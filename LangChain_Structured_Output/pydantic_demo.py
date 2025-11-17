from pydantic import BaseModel, Field , EmailStr
from typing import Optional , Literal 
class Student(BaseModel):
    name : str = 'alice'
    age :Optional[int] = None
    email : EmailStr 
    cgpa : float = Field(gt=0 , lt= 4 , description="CGPA must be between 0 and 4")
new_student = {'age': 20, 'email': 'abc@gmail.com' , 'cgpa': 3.5}
student = Student(**new_student)

print(student) # name='ali' age=<class 'pydantic.fields.UndefinedType'>
print(student.name) # alice