import time
import sys

def user_input_process():
    while True:
        user_input = input("\nEnter new desired position as 'x y z' (or 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting user input process.")
            break
        with open("desired_position.txt", "w") as f:
            f.write(user_input)
        print(f"Updated desired position to: {user_input}")

if __name__ == "__main__":
    user_input_process()
