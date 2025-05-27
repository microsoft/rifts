# this is gonna be the script that highlights examples of evaluating on CollaBench


tasks = [
    "What is the capital of France?",
]


def evaluate_model(model):
    for task in tasks:
        print(task)
        print(model.predict(task))
        print()
