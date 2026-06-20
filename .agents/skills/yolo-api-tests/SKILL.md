---
name: yolo-api-tests
description: Use this skill when writing API tests for the YOLO service endpoints. Generate HTTP-level tests using FastAPI TestClient, temporary SQLite databases, and mocked YOLO models.

# YOLO API Testing Skill

When writing tests for the YOLO service:
  - Use pytest (or unittest when explicitly requested).
  - Test the HTTP API endpoints, not internal functions directly.
  - Use FastAPI TestClient.
  - Use a temporary SQLite database for every test.
  - Never use the real production database.
  - Mock the YOLO model and any heavy dependencies.
  - Assert both the HTTP status code and the response body structure.
  - Validate response fields and expected data types.
  - Name test files with the prefix `test_`.
  - Prefer isolated and repeatable tests.
  - Clean up temporary files after test execution.
  - Optionally use Pydantic models to validate response schemas.

Example expectations:
  - Status code is verified.
  - Response JSON contains expected fields.
  - Error cases are tested.
  - No real model inference is executed.
---
