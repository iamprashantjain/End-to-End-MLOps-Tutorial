#initiate a class
class Employee:
    #special method/dunder method -- constructor
    def __init__(self):
        print("init called")
        self.name = "Sam Altman"
        self.id = 123
        self.salary = 5000
        self.designation = "SDE"
        

    #function defined inside a class is called method
    def report(self, what):
        print("report called")
        return(f"Im reporting: {what}")
        
    def coding(self, what):
        print("coding called")
        return(f"Im coding in: {what}")
       
    
#creating instance of class
sam = Employee()
print(sam.name)
print(sam.id)

print(sam.coding("Python"))

sam.dob = "26Jan"
print(sam.dob)