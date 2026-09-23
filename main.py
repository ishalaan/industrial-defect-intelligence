from modules import test

from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

def main():
    #Tests
    print(f"Test ENV Value: {os.getenv('TEST_ENV_VAR')}")
    test.test_function("Hello World!")

if __name__ == "__main__":
    main()