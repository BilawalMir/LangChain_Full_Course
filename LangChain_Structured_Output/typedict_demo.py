from typing import TypedDict

class person(TypedDict):
    name: str
    age: int

new_person:person = {'name': 'alice'}
print(new_person)