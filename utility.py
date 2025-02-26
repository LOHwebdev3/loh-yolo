import random
import string
import re

def to_snake_case(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)  # Add underscore before uppercase letters
    return text.lower()


def generate_unique_string(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

