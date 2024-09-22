class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience (int): Numero di epoche da attendere prima di fermarsi se non ci sono miglioramenti.
            min_delta (float): Miglioramento minimo richiesto per considerare un cambiamento come significativo.
            mode (str): Se 'min', si aspetta una diminuzione del valore monitorato (adatto per la loss).
                        Se 'max', si aspetta un aumento (adatto per accuracy o F1-score).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_score is None:
            self.best_score = metric_value
        else:
            if self.mode == 'min':  # Per regressione (monitoraggio della loss)
                if metric_value < self.best_score - self.min_delta:
                    self.best_score = metric_value
                    self.counter = 0  # Resetta il contatore quando trovi un miglioramento
                else:
                    self.counter += 1
            elif self.mode == 'max':  # Per classificazione (monitoraggio di accuracy o F1-score)
                if metric_value > self.best_score + self.min_delta:
                    self.best_score = metric_value
                    self.counter = 0  # Resetta il contatore quando trovi un miglioramento
                else:
                    self.counter += 1

        # Fermati se la pazienza è stata superata
        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

