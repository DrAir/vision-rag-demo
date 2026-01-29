from unittest.mock import MagicMock
import os
import shutil
# Import app to test its logic
import app

# Mock Gradio file object
class MockFile:
    def __init__(self, name):
        self.name = name

def test_process_pdf():
    print("Test 1: Testing with Mock File Object...")
    # Create a dummy PDF file for testing if not exists (using one from temp dir if available or just a dummy file)
    # Ideally we need a real PDF. Let's look for one.
    # Since we don't have a guaranteed PDF, we will mock convert_from_path to avoid needing a real PDF, 
    # OR we can just try with a dummy path and expect a specific error if file invalid, 
    # BUT we want to test the PATH extraction logic mainly.
    
    # Let's mock the heavy dependencies to test JUST the path logic
    app.convert_from_path = MagicMock(return_value=[]) 
    app.RAG_MODEL = MagicMock()
    app.RAG_MODEL.index = MagicMock()
    
    dummy_path = "/tmp/dummy.pdf"
    mock_file = MockFile(dummy_path)
    
    result = app.process_pdf(mock_file)
    print(f"Result 1: {result}")
    
    # Assertions
    # Check if convert_from_path was called with the correct path
    app.convert_from_path.assert_called_with(dummy_path)
    print("Assertion 1 Passed: convert_from_path called with correct path from object.")

    print("\nTest 2: Testing with String Path...")
    app.convert_from_path.reset_mock()
    
    result2 = app.process_pdf(dummy_path)
    print(f"Result 2: {result2}")
    
    app.convert_from_path.assert_called_with(dummy_path)
    print("Assertion 2 Passed: convert_from_path called with correct path from string.")

if __name__ == "__main__":
    test_process_pdf()
