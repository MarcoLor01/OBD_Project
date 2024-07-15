class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_score is None:
            self.best_score = metric_value
        elif metric_value > self.best_score + self.min_delta:
            self.counter += 1
        else:
            self.best_score = min(metric_value, self.best_score)
        if self.counter == self.patience:
            self.early_stop = True
        return self.early_stop

