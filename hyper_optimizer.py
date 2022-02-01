from comet_ml import Experiment, Optimizer

API_KEY = 'COMET_API_KEY'
optimizer = Optimizer(API_KEY)

params = """
x integer [1, 10] [10]
y real [1, 10] [1.0]
"""

while True:
    suggestion = optimizer.get_suggestion()
    experiment = Experiment("API_KEY")
    score = fit(suggestion["x"])
    suggestion.report_score("accuracy", score)

    
    
