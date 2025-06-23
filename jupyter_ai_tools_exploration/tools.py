import logging
from typing import Literal

logger = logging.getLogger(__name__)

def add_cell(
        file_id: str, 
        content: str | None = None,
        cell_index: int | None = None,
        add_above: bool = False,
        cell_type: Literal["code", "markdown", "raw"] = "code"
    ):
    """Adds a new cell to the notebook above or below a specified cell index"""
    logger.info(
        f"Added a new cell to notebook " 
        f"{file_id=}, {content=}, {cell_index=}, {add_above=}, {cell_type=}"
    )

def delete_cell(file_id: str, cell_index: int):
    """Removes a notebook cell at the specified cell index"""
    logger.info(f"Deleted cell at {cell_index=} from notebook {file_id=}")

def _is_notebook_active(file_id: str) -> bool:
    """Returns True if the document is active"""
    pass

def _get_notebook_source(file_id: str) -> str:
    """Returns the notebook source as json"""
    pass

def _get_notebook_source_md(file_id: str) -> str:
    """Returns the notebook source as markdown string"""
    pass
