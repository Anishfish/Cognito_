def load_to_csv1(fname='/Users/Sumeet/Desktop/CognitoMaster/user.txt'):
        age = input("Please enter age: ")
        age = int(age)
        hand = input("Please select r for right, l for left, or n for none for your hand: ")
        gender = input("Please enter gender: m for male and f for female")
        if(hand == 'r'):
            hand == 'Right'
        elif(hand == 'l'):
            hand = "Left"
        else:
            hand = "None"
        if(gender == "m"):
            gender= "Male"
        elif(gender == "f"):
            gender = "Female"
        else:
            gender = "None"
        with open(fname, 'a') as f:
            mansssssss = ("age = {}\nhand = {}\ngender = {}").format(age, hand, gender)
            f.write(mansssssss)

load_to_csv1()
