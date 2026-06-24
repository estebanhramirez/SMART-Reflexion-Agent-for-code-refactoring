# General imports
import os
import re # To parse the GitHub's responses and extract modified files.
import json # To parse the LLM's responses and extract the tool calls.
import subprocess # To execute the GitHub commands and capture their output.
from pathlib import Path # Used for file handling and directory listing
from typing import Any, Dict, Optional, List, Sequence # Used for type hinting the tools' inputs and outputs

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
from langchain_groq import ChatGroq # To create the LLM that will drive the agent's reasoning and tool usage.
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


# =========================================== BEGGINING OF EXECUTION ===========================================


class Reflection(BaseModel):
    effects: str = Field(description="What are the possible effects of executing the function calls in this instructions")
    conflicts: str = Field(description="What are the possible conflicts that may arise from executing the function calls in this instructions")
    needed: str = Field(description="What other function calls might still be needed to achieve the final goal")
    unneeded: str = Field(description="What function calls might not be needed to achieve the final goal")

class Instructions(BaseModel):
    explanation: str = Field(description="How the function calls in this instructions help achieve the final goal")
    reflection: Reflection = Field(description="Self-critique of the sequence of functions")
    function_calls: List[Dict[str, str]] = Field(description="Sequence of function calls (dictionary with keys `func_name` and `args`) to be executed towards the realization of the final goal")


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]: 
    tool_messages = []
    last_message = state[-1]

    print(last_message)
    print("===============================================================\n")

    tools_map = {
        "get_modified_files": get_modified_files,
        "get_file_changes": get_file_changes,
        "get_file_content": get_file_content,
        "list_files_in_dir": list_files_in_dir,
        "agent_modify_line": agent_modify_line
    }

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "Instructions":
            instructions_tool_calls = tool_call["args"].get("instructions_tool_calls", [])

            for instructions_tool_call in instructions_tool_calls:
                print(instructions_tool_call)
                # Execute the sub-tool
                result = tools_map[instructions_tool_call["fuction_name"]].invoke(instructions_tool_call["args"])

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        else:
            # Handle normal tool calls if they happen outside of "Instructions"
            result = tools_map[tool_call["name"]].invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
    return (tool_messages)


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    print(f"---------------------> {num_iterations}")
    if num_iterations >= 5:
        return END
    return "execute_tools"



if __name__ == "__main__":

    # LLMs definition
    import os

    groq_api_key = os.getenv("GROQ_API_KEY")
    gcp_api_key = os.getenv("GCP_API_KEY")

    os.environ["GROQ_API_KEY"] = groq_api_key  

    llm_responder_langchain_workflow = ChatGroq(model="llama-3.1-8b-instant", temperature=0.6)
    # ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gcp_api_key)
    llm_revisor_langchain_workflow = ChatGroq(model="llama-3.1-8b-instant", temperature=0.6)
    # ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gcp_api_key)

    # Prompt template definition
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Reflexion Agent that can reason about the changes to be made in a GitHub repository.

            You must propose a plan by calling the 'Instructions' tool provided to you. Inside this tool, you can schedule an ordered sequence of operations using the following supported function names and arguments:
                
                get_modified_files
                    Args: None
                    Returns: List[str]

                get_file_changes
                    Args: modified_files (List[str])
                    Returns: str

                get_file_content
                    Args: file_path (str), encoding (str)
                    Returns: str

                list_files_in_dir
                    Args: folder_path (str)
                    Returns: List[str]

                agent_modify_line
                    Args: file_path (str), line_number (int), new_content (str)
                    Returns: Optional[str]

            Follow these steps:
                1. {instruction}
                2. Think how the operations above can help achieve the final goal.
                3. Reflect about the possible consequences of executing each operation.
                4. Populate the 'Instructions' tool with the exact sequence required.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "You must respond by executing the 'Instructions' structured tool mapping out your next steps."
        ),
    ])

    # Nodes logic definition
    responder_instruction = ""
    responder_langchain_workflow = prompt_template.partial(instruction = responder_instruction) | llm_responder_langchain_workflow.bind_tools(tools=[Instructions])

    revisor_instruction = ""
    revisor_langchain_workflow = prompt_template.partial(instruction = revisor_instruction) | llm_revisor_langchain_workflow.bind_tools(tools=[Instructions])

    # Graph definition
    graph = MessageGraph()

    graph.add_node("respond", responder_langchain_workflow)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revisor", revisor_langchain_workflow)

    graph.set_entry_point("respond")

    graph.add_edge("respond", "execute_tools")
    graph.add_edge("execute_tools", "revisor")
    graph.add_conditional_edges("revisor", event_loop)

    # Execute the agent
    app = graph.compile()
    responses = app.invoke(
        """
            Add a `return(0)` statement at the bottom of the main function of the file test.py, without modifying the rest of the code.
        """
    )

    # Visualize the messages history
    print(f"\n\nNumber of messages in the history: {len(responses)} (See details below)\n")
    for i, message in enumerate(responses):
        print(f"Message No. {i}: {message}\n")