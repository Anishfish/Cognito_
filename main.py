from sendemail import confirmUserEmail
from keypress import k_press 
from run import analysis, finalEmail
from user import load_to_csv1


if __name__ == '__main__':
    confirmUserEmail()
    load_to_csv1()
    k_press()
    analysis()
    finalEmail()
