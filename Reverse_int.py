class INTEGER:
    is_NEG = False
    val = 0
    def __init__(self,Value,NEG_POS):
        self.is_NEG = NEG_POS
        self.val = Value

def Reverse_int(Target):
    
    if Target.val > 2**31 - 1 :
        return 0
    elif Target.val < -2**31 :
        return 0

    A = " "+str(Target.val)
    num = len(A)
    B = ""
    for i in range(1,num):
        if A[-i] == 0 :
            continue
        else :
            B += A[-i]
    if Target.is_NEG :
        return int("-" + B)
    else :
        return int(B)

Input_data = int(input(""))
Is_M = False
if Input_data > 0 :
    Is_M = False
else :
    Input_data = Input_data*(-1)
    Is_M = True

A = INTEGER(Input_data,Is_M)
print(Reverse_int(A))
