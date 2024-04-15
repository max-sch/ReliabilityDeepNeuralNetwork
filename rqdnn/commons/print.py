def print_start(model_name):
    print("".center(70, '='))
    print("Model: " + model_name)

def print_end():
    print("".center(70, '='))

def print_progress(message):
    print(message.center(70, '-'))

def print_result(metric, values):
    print("{metric}: {values}".format(metric=metric, values=values))