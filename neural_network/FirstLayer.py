class FirstLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs

