def print_progress(message):
    print(message.center(60, '#'))

def print_result(metric, values):
    print("{metric}: {values}".format(metric=metric, values=values))