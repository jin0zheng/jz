from time import sleep
def test_answer(cmdopt):
    if cmdopt == "type1":
        print("first")
    elif cmdopt == "type2":
        print("second")
    else:
        print(cmdopt.center(100, '='))
    assert 1
