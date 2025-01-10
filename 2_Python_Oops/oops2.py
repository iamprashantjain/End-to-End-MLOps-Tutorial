class Social:
    def __init__(self):
        self.username = ""
        self.password = ""
        self.loggedin = False
        
        # calling menu function as soon as object is created
        self.menu()
        
        
    def menu(self):
        user_input = input("""" Welcome to Social, Choose Options below!
                           
                           1. Press 1 to Signup
                           2. Press 2 to Signin
                           3. Press 3 to Write Post
                           5. Press any key to Exit
                           
                           """)
        
        if user_input == "1":
            self.signup()
        
        elif user_input == "2":
            self.signin()
        
        elif user_input == "3":
            self.writepost()
              
        else:
            print("See You Again!")
            exit()
            
    
    def signup(self):
        email = input("Enter Email: ")
        password = input("Enter Password: ")
        
        #asign email and password in constructor function
        self.username = email.lower()
        self.password = password.lower()
        print("Signup Done")
        print("\n")
        
        #display the menu again
        self.menu()
        
        
        
    def signin(self):
        email = (input("Enter Email: ")).lower()
        password = (input("Enter Password: ")).lower()
        
        # Check if entered credentials match
        if email == self.username and password == self.password:
            #setting the flag to true
            self.loggedin = True
            print("Signin Successful")
            self.menu()
        else:
            print("Invalid email or password")
            retry = input("Do you want to try again? (y/n): ")
            if retry.lower() == 'y':
                self.signin()
            else:
                print("Returning to main menu...")
                self.menu()      
            
    
    def writepost(self):
        if self.loggedin == True:
            post = input("Write your post...")
            print()
            print(f"Your Post: {post}")              
        
        else:
            print("Signin is Required!")
            
            action = input("Press 1 for Signin or 2 for Signup")
                        
            if action == "1":
                self.signin()
                
            if action == "2":
                self.signup()

            
obj = Social()