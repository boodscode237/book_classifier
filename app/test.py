import requests
import json
import os
import time
from pathlib import Path
import pytest
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # Path to a test image


class BookPageClassifierAPITest:
    """Test suite for Book Page Classifier API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_image_id = None

    def test_health_check(self):
        """Test health check endpoint"""
        print("Testing health check...")
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        print("✅ Health check passed")
        return True

    def test_predict_endpoint(self, image_path: str):
        """Test image prediction endpoint"""
        print(f"Testing prediction with image: {image_path}")

        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return False

        # Open and send image
        with open(image_path, "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = self.session.post(f"{self.base_url}/predict", files=files)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = ["image_id", "predicted_class", "class_name", "message"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Validate data types and ranges
        assert isinstance(data["predicted_class"], int)
        assert 0 <= data["predicted_class"] <= 3
        assert isinstance(data["class_name"], str)
        assert isinstance(data["image_id"], str)

        # Store image ID for subsequent tests
        self.test_image_id = data["image_id"]

        print(
            f"✅ Prediction successful: {data['class_name']} (class {data['predicted_class']})"
        )
        print(f"   Image ID: {data['image_id']}")
        return True

    def test_get_classification(self):
        """Test get classification endpoint"""
        if not self.test_image_id:
            print("❌ No test image ID available")
            return False

        print(f"Testing get classification for image: {self.test_image_id}")
        response = self.session.get(
            f"{self.base_url}/classification/{self.test_image_id}"
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = ["image_id", "model_class", "model_class_name", "created_at"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        assert data["image_id"] == self.test_image_id
        assert isinstance(data["model_class"], int)
        assert 0 <= data["model_class"] <= 3

        print("✅ Get classification successful")
        return True

    def test_correct_classification(self):
        """Test classification correction endpoint"""
        if not self.test_image_id:
            print("❌ No test image ID available")
            return False

        print(f"Testing classification correction for image: {self.test_image_id}")

        # Correct classification to class 1
        correction_data = {"user_class": 1}
        response = self.session.put(
            f"{self.base_url}/correct/{self.test_image_id}", data=correction_data
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response
        assert data["image_id"] == self.test_image_id
        assert data["user_class"] == 1
        assert data["class_name"] == "Title Page"

        print("✅ Classification correction successful")
        return True

    def test_get_stats(self):
        """Test statistics endpoint"""
        print("Testing statistics endpoint...")
        response = self.session.get(f"{self.base_url}/stats")

        assert response.status_code == 200
        data = response.json()

        assert "statistics" in data
        assert isinstance(data["statistics"], list)

        # Validate statistics structure
        for stat in data["statistics"]:
            required_fields = [
                "class",
                "class_name",
                "total_predictions",
                "user_corrections",
            ]
            for field in required_fields:
                assert field in stat

        print("✅ Statistics endpoint successful")
        return True

    def test_invalid_image_id(self):
        """Test with invalid image ID"""
        print("Testing with invalid image ID...")

        invalid_id = "invalid-uuid"
        response = self.session.get(f"{self.base_url}/classification/{invalid_id}")
        assert response.status_code == 400

        print("✅ Invalid image ID handled correctly")
        return True

    def test_nonexistent_image_id(self):
        """Test with nonexistent image ID"""
        print("Testing with nonexistent image ID...")

        nonexistent_id = "00000000-0000-0000-0000-000000000000"
        response = self.session.get(f"{self.base_url}/classification/{nonexistent_id}")
        assert response.status_code == 404

        print("✅ Nonexistent image ID handled correctly")
        return True

    def test_invalid_file_type(self):
        """Test with invalid file type"""
        print("Testing with invalid file type...")

        # Create a dummy text file
        dummy_content = b"This is not an image"
        files = {"file": ("test.txt", dummy_content, "text/plain")}
        response = self.session.post(f"{self.base_url}/predict", files=files)

        assert response.status_code == 400

        print("✅ Invalid file type handled correctly")
        return True

    def test_invalid_user_class(self):
        """Test correction with invalid user class"""
        if not self.test_image_id:
            print("❌ No test image ID available")
            return False

        print("Testing correction with invalid user class...")

        # Try to correct with invalid class
        correction_data = {"user_class": 5}  # Invalid class
        response = self.session.put(
            f"{self.base_url}/correct/{self.test_image_id}", data=correction_data
        )

        assert response.status_code == 400

        print("✅ Invalid user class handled correctly")
        return True

    def run_load_test(self, image_path: str, num_requests: int = 10):
        """Run load test with multiple concurrent requests"""
        print(f"Running load test with {num_requests} requests...")

        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return False

        import concurrent.futures
        import time

        def make_request():
            with open(image_path, "rb") as f:
                files = {"file": ("test_image.jpg", f, "image/jpeg")}
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files)
                end_time = time.time()
                return response.status_code, end_time - start_time

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Analyze results
        successful_requests = sum(1 for status, _ in results if status == 200)
        response_times = [time for _, time in results]
        avg_response_time = sum(response_times) / len(response_times)

        print(f"✅ Load test completed:")
        print(f"   Successful requests: {successful_requests}/{num_requests}")
        print(f"   Average response time: {avg_response_time:.2f}s")

        return successful_requests == num_requests

    def run_all_tests(self, image_path: str = TEST_IMAGE_PATH):
        """Run all tests"""
        print("=" * 50)
        print("BOOK PAGE CLASSIFIER API TEST SUITE")
        print("=" * 50)

        tests = [
            ("Health Check", lambda: self.test_health_check()),
            ("Prediction Endpoint", lambda: self.test_predict_endpoint(image_path)),
            ("Get Classification", lambda: self.test_get_classification()),
            ("Correct Classification", lambda: self.test_correct_classification()),
            ("Statistics Endpoint", lambda: self.test_get_stats()),
            ("Invalid Image ID", lambda: self.test_invalid_image_id()),
            ("Nonexistent Image ID", lambda: self.test_nonexistent_image_id()),
            ("Invalid File Type", lambda: self.test_invalid_file_type()),
            ("Invalid User Class", lambda: self.test_invalid_user_class()),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
                    print(f"❌ {test_name} failed")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name} failed with error: {e}")

        print("\n" + "=" * 50)
        print(f"TEST RESULTS: {passed} passed, {failed} failed")
        print("=" * 50)

        # Run load test if all basic tests passed
        if failed == 0:
            print("\n--- Load Test ---")
            self.run_load_test(image_path, 5)

        return failed == 0


def create_test_image():
    """Create a simple test image if none exists"""
    if not os.path.exists(TEST_IMAGE_PATH):
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create a simple test image
            img = Image.new("RGB", (224, 224), color="white")
            draw = ImageDraw.Draw(img)

            # Add some text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), "Test Book Page", fill="black", font=font)
            draw.text((10, 50), "Sample Text", fill="black", font=font)
            draw.rectangle([10, 80, 200, 120], outline="black", width=2)

            img.save(TEST_IMAGE_PATH)
            print(f"Created test image: {TEST_IMAGE_PATH}")
            return True
        except Exception as e:
            print(f"Could not create test image: {e}")
            return False
    return True


if __name__ == "__main__":
    # Create test image if it doesn't exist
    create_test_image()

    # Run tests
    tester = BookPageClassifierAPITest()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
