# General imports
import re # To parse the GitHub's responses and extract modified files.
import subprocess # To execute the GitHub commands and capture their output.
from pathlib import Path # Used for file handling and directory listing
from typing import Optional, List, Sequence # Used for type hinting the tools' inputs and outputs

# LangChain imports
from langchain.tools import tool # To create tools through the @tool decorator.
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # To create the LCEL chains.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # To create the prompt templates for the LLM.

from langchain_core.messages import BaseMessage # To create a general human/tool/system messages (parent class).
from langchain_core.messages import HumanMessage # To create human messages with the user's query.
from langchain_core.messages import ToolMessage # To create tool messages with the tools' responses.
from langchain_core.messages import SystemMessage # To create system messages for the LLM.
from langchain_core.messages import AIMessage # To create AI messages with the LLM's responses.

# LangGraph imports
from langgraph.graph import END
from langgraph.graph import MessageGraph # To streamline workflow creation and state management: LangGraph provides the prebuilt solution `MessageGraph`; This simplifies the process of setting up conversational workflows by handling the underlying structure automatically.

# LLMs imports
from langchain_google_genai import ChatGoogleGenerativeAI # To create the LLM that will drive the agent's reasoning and tool usage.

# Pydantic imports
from pydantic import BaseModel, Field # To create the Pydantic models for the instructions and reflection.


# =========================================== GitHub Functions ===========================================


def run_git_status():
    try:
        result = subprocess.run( # Run the command and capture the output
            ["git", "status"], 
            capture_output=True,  # Captures stdout and stderr
            text=True,            # Automatically decodes bytes to a string
            check=True            # Raises an error if the command fails
        )
        git_status_string = result.stdout # 'result.stdout' contains the standard output as a normal Python string
        return (git_status_string)
    except subprocess.CalledProcessError as e:
        raise TypeError(f"Git error: {e.stderr}") # If the command fails, 'e.stderr' contains the error message as a string


def run_git_diff(modified_files):
    try:
        result = subprocess.run( # Run the command and capture the output
            ["git", "diff"] + modified_files, 
            capture_output=True,  # Captures stdout and stderr
            text=True,            # Automatically decodes bytes to a string
            check=True            # Raises an error if the command fails
        )
        git_diff_string = result.stdout # 'result.stdout' contains the standard output as a normal Python string
        return (git_diff_string)
    except subprocess.CalledProcessError as e:
        raise TypeError(f"Git error: {e.stderr}") # If the command fails, 'e.stderr' contains the error message as a string


# =========================================== Tools ===========================================

@tool
def get_modified_files() -> List[str]:
    """
        A function that retrieves the list of modified files in a GitHub repository by executing the 'git status' command and parsing its output.

        Args:
            None
        Returns:
            List[str]: A list of modified files in the GitHub repository.
    """
    git_status_output = run_git_status()
    pattern = r"modified:\s+(.+)"
    matches = re.findall(pattern, git_status_output) # Find all matches in the text
    modified_files = [match.strip() for match in matches] # Clean up any trailing whitespace from the file names
    return (modified_files)

@tool
def get_file_changes(modified_files: List[str]) -> str:
    """
        A function that retrieves the changes in the modified files of a GitHub repository by executing the 'git diff' command and parsing its output.

        Args:
            modified_files (List[str]): A list of modified files for which to retrieve changes.
        Returns:
            str: The output of the 'git diff' command.
    """
    git_diff_output = run_git_diff(modified_files)
    return (git_diff_output)

@tool
def get_file_content(file_path: str, encoding: str = "utf-8") -> str:
    """
        Opens a file and returns its content as a string.
      
        Args:
            file_path (str): The path to the file you want to read.
            encoding (str): The text encoding (defaults to 'utf-8').
        Returns:
            str: The file content as a string, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return (file.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file at '{file_path}' was not found.")
        return (None)
    except PermissionError:
        raise PermissionError(f"Error: Permission denied when accessing '{file_path}'.")
        return (None)
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"Error: Could not decode the file using {encoding} encoding.")
        return (None)

@tool
def list_files_in_dir(folder_path: str = ".") -> List[str]:
    """
        Lists all files in the specified folder (non-recursive).
        
        Args:
            folder_path (str): The path to the directory (defaults to current directory ".").
        Returns:
            List[str]: A list of file names as strings.
    """
    path = Path(folder_path)

    # Ensure the path exists and is actually a directory
    if not path.exists():
        raise FileNotFoundError(f"Error: The path '{folder_path}' does not exist.")
        return []
    if not path.is_dir():
        raise NotADirectoryError(f"Error: The path '{folder_path}' is a file, not a directory.")
        return []

    # Iterate through the directory and filter for files only
    file_list = [str(item.resolve()) for item in path.rglob("*") if item.is_file()]

    return (file_list)

@tool
def agent_modify_line(file_path: str, line_number: int, new_content: str) -> Optional[str]:
    """
        Modifies a specific line in a file based on a 1-indexed line number.
        
        Args:
            file_path (str): Path to the .txt or .py file.
            line_number (int): The line number to change (1-indexed, meaning line 1 is the first line).
            new_content (str): The new text string to place at that line.
        Returns:
            Optional[str]: A success message string, or an error message if it fails.
    """
    if line_number < 1:
        return "Error: Line numbers must be 1 or greater."
    try:
        with open(file_path, "r", encoding="utf-8") as file: # 1. Read all lines from the file
            lines = file.readlines()
        # 2. Check if the requested line number actually exists
        if line_number > len(lines):
            return f"Error: File only has {len(lines)} lines. Cannot modify line {line_number}."     
        # 3. Ensure the new content ends with a newline character so lines don't smash together
        if not new_content.endswith("\n"):
            new_content += "\n"       
        # 4. Modify the target line (subtract 1 because Python lists start at 0)
        lines[line_number - 1] = new_content
        # 5. Write the updated lines back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        return (f"Success: Modified line {line_number} in '{file_path}'.")
        
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except Exception as e:
        return f"Error modifying file: {str(e)}"



def execute_tool(tool_call, tools_map):
    print(f"Debugging: tool_call and tools_map exist. tool_call={tool_call}, tools_map={tools_map}")
    output_tool = tools_map[tool_call['name']].invoke(tool_call['args'])
    tool_message = ToolMessage(content=output_tool, tool_call_id=tool_call['id'])
    return (tool_message)


def recursive_tool_execution(messages_history, llm, tools_map):
    latest_message = messages_history[-1]

    if getattr(latest_message, 'tool_calls', False):
        for tool_call in latest_message.tool_calls:
            tool_message = execute_tool(tool_call, tools_map)
            messages_history.append(tool_message)

        llm_response = llm.invoke(messages_history)
        messages_history.append(llm_response)

        return(recursive_tool_execution(messages_history, llm, tools_map))
    else:
        return (messages_history)


# =========================================== BEGGINING OF EXECUTION ===========================================


class Reflection(BaseModel):
    effects: str = Field(description="What are the possible effects of executing the commands in this instructions")
    conflicts: str = Field(description="What are the possible conflicts that may arise from executing the commands in this instructions")
    needed: str = Field(description="What other commands might still be needed to achieve the final goal")
    unneeded: str = Field(description="What commands might not be needed to achieve the final goal")

class Instructions(BaseModel):
    objective: str = Field(description="How the commands in this instructions help achieve the final goal")
    reflection: Reflection = Field(description="Self-critique of the instructions")
    commands: List[str] = Field(description="Sequence of commands to be executed towards the realization of the final goal")



if __name__ == "__main__":

    # Initial node ===========================================================
    def propose_initial_changes_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
        """
            The `propose_initial_changes_node` function acts as the starting point in the Reflection Agent's
            workflow. It generates initial changes based on the current state of the conversation,
            which contains all previous messages (user inputs, AI responses, and system instructions).

            Args:
                state (Sequence[BaseMessage]): A sequence of BaseMessage objects (i.e., HumanMessage, AIMessage, SystemMessage, ToolMessage).
                                               These messages provide the context necessary for generating a meaningful response.

            Returns:
                List[AIMessage]: A list containing a single AIMessage object that encapsulates the proposed initial changes.
                                 The content of this message is derived from the output of the `initial_langchain_workflow`,
                                 which processes the current state of the conversation to generate a response.
        """

        initial_langchain_workflow_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key="AQ.Ab8RN6II3pE9opZ-ENdyg1dm8GB4XoOv0pU-dMTYd9eNMSCokA" # "AQ.Ab8RN6KUrq59ty82WP5wvxPe2jPbHxoDmXzxph_F8E_qoFcTpg"
        )

        tools = [get_modified_files, get_file_changes, get_file_content, list_files_in_dir, agent_modify_line]
        tools_map = { tool.name:tool for tool in tools }

        initial_langchain_workflow_llm_with_tools = initial_langchain_workflow_llm.bind_tools(tools)

        # 2. We use ChatPromptTemplate from LangChain to structure the prompt. The prompt has two main parts:
        # 2.1. A SystemMessage that sets the context for the LLM, explaining its role as a Reflection Agent.
        # 2.2. A MessagesPlaceholder object used to inject the actual content or message that the post will be based on. The placeholder will be populated with the user’s request at runtime.
        initial_langchain_workflow_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a Reflection Agent that can reason about the changes to be made in a GitHub repository. "
                "You have access to the following tools: get_modified_files, get_file_changes, get_file_content, "
                "list_files_in_dir, and agent_modify_line. Use these tools to gather information and propose changes.",
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

        # 3. Restructure the Workflow Chain
        initial_langchain_workflow = (
            # Step A: Format the prompt, then instantly invoke the LLM
            RunnablePassthrough.assign(
                messages = lambda x: initial_langchain_workflow_llm_with_tools.invoke(
                    initial_langchain_workflow_prompt.format_messages(messages=x['messages'])
                )
            )
            |
            # Step B: Append the newly generated AI message to the historic list
            RunnablePassthrough.assign(
                messages = lambda x: x['messages'] + [x['messages']] # Combines old list + newest LLM response
            )
            |
            # Step C: Hand off the accumulated history to your recursive tool executor
            RunnablePassthrough.assign(
                messages = lambda x: recursive_tool_execution(x['messages'], initial_langchain_workflow_llm_with_tools, tools_map)
            )
        )

        # 4. Invoke the chain
        output_state = initial_langchain_workflow.invoke({"messages": state})
        
        # 5. Correctly extract the last AIMessage from the returned dictionary list
        final_msg = output_state["messages"][-1]
        
        return [final_msg]


        """
            A sequence of messages representing the current state of the conversation;
            includes previous AI responses, user inputs, and system-level instructions.
            The messages are used to provide context to the reflection process, guiding
            the generation of a more refined output.
        """




    graph = MessageGraph()



    # Execute the agent
    initial_query = {'content': "Add a debugging print statement inside the `execute_tool()` function in the file `main.py`, to make sure the arguments to the function exists."}
    initial_langchain_workflow_output = initial_langchain_workflow.invoke(initial_query)

    # Visualize the messages history
    print(f"\n\nNumber of messages in the history: {len(initial_langchain_workflow_output['messages_history'])} (See details below)\n")
    for i, message in enumerate(initial_langchain_workflow_output['messages_history']):
        print(f"Message No. {i}: {message}\n")