import pytest
import logging
import os
import sys
from unittest.mock import patch, mock_open
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional, Set, Any

# ------------------- Logging Configuration -------------------

# Setting up the log folder and file naming convention
# Check if the 'logs' folder exists, if not create it
log_folder = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Get the current test file name without extension to use in the log file name
log_file_name = os.path.splitext(os.path.basename(__file__))[0] + "_test_logs.log"
log_file_path = os.path.join(log_folder, log_file_name)

# Configuring logging to write log messages to the specified file
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,  # Setting log level to INFO, so INFO and higher level messages are logged
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of the log messages
    force=True  # Ensure that logging is reset each time this file is run
)

# Add the path to the scripts module to ensure it is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.scraper import scrape_wikipedia_page, extract_valid_links, auto_crawl_and_save_one_file


# ------------------- scrape_wikipedia_page -------------------

@patch("scripts.scraper.requests.get")
def test_scrape_success(mock_get: Any) -> None:
    """
    Test case for successful scraping of a Wikipedia page.
    
    Mocks a successful HTTP GET request and checks that the
    content is correctly extracted.
    
    Args:
        mock_get: Mocked requests.get function
        
    Asserts:
        1. The text extracted contains expected content
        2. The soup object is an instance of BeautifulSoup
    """
    # Mocking the HTML response for the Wikipedia page
    html = '''
    <html><div class="mw-parser-output">
    <p>First paragraph.</p><p>Second paragraph.</p></div></html>
    '''
    mock_get.return_value.status_code = 200  # Simulate successful HTTP response
    mock_get.return_value.text = html  # Set the mock response text (HTML content)

    # Calling the function to test
    text, soup = scrape_wikipedia_page("https://en.wikipedia.org/wiki/Test")
    
    # Check that the text extraction is correct
    assert "First paragraph." in text
    assert isinstance(soup, BeautifulSoup)

    # Logging the test result
    logging.info("test_scrape_success passed")
    # Flush the log buffer to ensure it is written immediately
    logging.getLogger().handlers[0].flush()


@patch("scripts.scraper.requests.get")
def test_scrape_network_error(mock_get: Any) -> None:
    """
    Test case for handling network errors during scraping.
    
    Mocks a network error (e.g., timeout, connection error) and verifies
    that the function handles it gracefully.
    
    Args:
        mock_get: Mocked requests.get function
        
    Asserts:
        Both return values are None when a network error occurs
    """
    # Simulating a network error by raising an exception
    mock_get.side_effect = Exception("Network error")
    
    # Calling the function to test
    text, soup = scrape_wikipedia_page("https://en.wikipedia.org/wiki/Test")
    
    # Check that the return values are None in case of error
    assert text is None
    assert soup is None
    
    # Logging the test result
    logging.info("test_scrape_network_error passed")
    logging.getLogger().handlers[0].flush()


@patch("scripts.scraper.requests.get")
def test_scrape_invalid_structure(mock_get: Any) -> None:
    """
    Test case for handling invalid HTML structure during scraping.
    
    Mocks a situation where the page structure does not match expectations
    and verifies the function returns None values.
    
    Args:
        mock_get: Mocked requests.get function
        
    Asserts:
        Both return values are None when the HTML structure is invalid
    """
    # Mocking an invalid HTML structure response
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "<html><body>No content div</body></html>"
    
    # Calling the function to test
    text, soup = scrape_wikipedia_page("https://en.wikipedia.org/wiki/Test")
    
    # Check that the function returns None due to invalid structure
    assert text is None
    assert soup is None
    
    # Logging the test result
    logging.info("test_scrape_invalid_structure passed")
    logging.getLogger().handlers[0].flush()


# ------------------- extract_valid_links -------------------

@pytest.mark.parametrize("html,expected", [
    # Test case for extracting valid links from HTML content
    (
        '''
        <a href="/wiki/Valid_Link">Valid</a>
        <a href="/wiki/File:Image">File</a>
        <a href="/wiki/Page#Section">Fragment</a>
        <a href="https://external.com">External</a>
        ''',
        ["https://en.wikipedia.org/wiki/Valid_Link"]
    ),
    # Test case where no valid links are present
    (
        "<html><body>No links</body></html>",
        []
    )
])
def test_extract_valid_links(html: str, expected: List[str]) -> None:
    """
    Test case for the function `extract_valid_links`, which extracts valid
    internal Wikipedia links from a page's HTML content.
    
    Args:
        html: HTML content to parse for links
        expected: List of expected valid links to be extracted
        
    Asserts:
        The set of extracted links matches the set of expected links
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Calling the function to extract valid links
    links = extract_valid_links(soup)
    
    # Assert that the extracted links match the expected list
    assert set(links) == set(expected)
    
    # Logging the test result
    logging.info("test_extract_valid_links passed")
    logging.getLogger().handlers[0].flush()


# ------------------- auto_crawl_and_save_one_file -------------------

@patch("scripts.scraper.scrape_wikipedia_page")
@patch("scripts.scraper.extract_valid_links")
@patch("scripts.scraper.open", new_callable=mock_open)
@patch("scripts.scraper.time.sleep", return_value=None)
def test_crawl_success(
    mock_sleep: Any, 
    mock_file: Any, 
    mock_links: Any, 
    mock_scrape: Any
) -> None:
    """
    Test case for successful crawling of Wikipedia pages and saving the result to a file.
    
    Mocks scraping, link extraction, and file writing operations to test
    the full crawling and saving workflow.
    
    Args:
        mock_sleep: Mocked time.sleep function
        mock_file: Mocked open function for file handling
        mock_links: Mocked extract_valid_links function
        mock_scrape: Mocked scrape_wikipedia_page function
        
    Asserts:
        1. File writing was called
        2. Scrape function was called once
        3. Link extraction function was called once
    """
    soup = BeautifulSoup("<div></div>", "html.parser")
    mock_scrape.return_value = ("Sample Content", soup)  # Mocking a successful scrape
    mock_links.return_value = []  # No links to extract
    
    # Calling the function to crawl and save data
    auto_crawl_and_save_one_file(
        ["https://en.wikipedia.org/wiki/Test_Page"], max_pages=1, output_file="output.txt"
    )

    handle = mock_file()  # Get the mock file handler
    handle.write.assert_called()  # Check that the file was written to
    mock_scrape.assert_called_once()  # Ensure that the scrape function was called once
    mock_links.assert_called_once()  # Ensure that the link extraction function was called once
    
    # Logging the test result
    logging.info("test_crawl_success passed")
    logging.getLogger().handlers[0].flush()


@patch("scripts.scraper.scrape_wikipedia_page", return_value=(None, None))
@patch("scripts.scraper.open", new_callable=mock_open)
@patch("scripts.scraper.time.sleep", return_value=None)
def test_crawl_with_error(
    mock_sleep: Any, 
    mock_file: Any, 
    mock_scrape: Any
) -> None:
    """
    Test case for handling errors during the crawling process.
    
    Mocks a failed scrape scenario and ensures the file write operation
    is not called when scraping fails.
    
    Args:
        mock_sleep: Mocked time.sleep function
        mock_file: Mocked open function for file handling
        mock_scrape: Mocked scrape_wikipedia_page function that returns None values
        
    Asserts:
        File write operation is not called when scraping fails
    """
    # Calling the function with a bad URL that fails to scrape
    auto_crawl_and_save_one_file(
        ["https://en.wikipedia.org/wiki/Bad_Page"], max_pages=1, output_file="fail.txt"
    )
    
    handle = mock_file()  # Get the mock file handler
    handle.write.assert_not_called()  # Ensure that the file was not written to due to error
    
    # Logging the test result
    logging.info("test_crawl_with_error passed")
    logging.getLogger().handlers[0].flush()
