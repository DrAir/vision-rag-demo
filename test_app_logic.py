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
    
    # Let's mock the heavy dependencies to test JUST the path logic
    app.convert_from_path = MagicMock(return_value=[]) 
    app.VLM_MODEL = MagicMock()
    
    dummy_path = "/tmp/dummy.pdf"
    mock_file = MockFile(dummy_path)
    
    # process_pdf now returns (status, gallery) tuple
    status, gallery = app.process_pdf(mock_file)
    print(f"Result 1: status={status}, gallery_len={len(gallery)}")
    
    # Assertions
    # Check if convert_from_path was called with the correct path
    app.convert_from_path.assert_called_with(dummy_path, dpi=150)
    print("Assertion 1 Passed: convert_from_path called with correct path from object.")

    print("\nTest 2: Testing with String Path...")
    app.convert_from_path.reset_mock()
    
    status2, gallery2 = app.process_pdf(dummy_path)
    print(f"Result 2: status={status2}, gallery_len={len(gallery2)}")
    
    app.convert_from_path.assert_called_with(dummy_path, dpi=150)
    print("Assertion 2 Passed: convert_from_path called with correct path from string.")

if __name__ == "__main__":
    test_process_pdf()
