from dataclasses import dataclass, fields, asdict


@dataclass
class ImageClassificationMetric:
    clean_correct: int = 0
    clean_total: int = 0
    clean_confidence: float = 0.0
    clean_loss: float = 0.0

    backdoor_correct: int = 0
    backdoor_total: int = 0
    backdoor_confidence: float = 0.0
    backdoor_loss: float = 0.0

    backdoored_correct: int = 0
    backdoored_total: int = 0
    backdoored_confidence: float = 0.0
    backdoored_loss: float = 0.0

    def update_from_batch(self, loss: float, correct: int, batch_size: int, confidence: float, type: str):
        if type == "clean":
            self.clean_loss = (self.clean_loss * self.clean_total + loss) / (batch_size + self.clean_total)
            self.clean_confidence = (self.clean_confidence * self.clean_total + confidence) / (batch_size + self.clean_total)
            self.clean_correct += correct
            self.clean_total += batch_size # This should be the last entry
        elif type == "backdoored":
            self.backdoored_loss = (self.backdoored_loss * self.backdoored_total + loss) / (batch_size + self.backdoored_total)
            self.backdoored_confidence = (self.backdoored_confidence * self.backdoored_total + confidence) / (batch_size + self.backdoored_total)
            self.backdoored_correct += correct
            self.backdoored_total += batch_size # This should be the last entry
        elif type == "backdoor":
            self.backdoor_loss = (self.backdoor_loss * self.backdoor_total + loss) / (batch_size + self.backdoor_total)
            self.backdoor_confidence = (self.backdoor_confidence * self.backdoor_total + confidence) / (batch_size + self.backdoor_total)
            self.backdoor_correct += correct
            self.backdoor_total += batch_size # This should be the last entry
        else:
            raise ValueError(f"Unknown metric type: {type}")

    @property
    def clean_accuracy(self) -> float:
        if self.clean_total == 0:
            return 0
        return self.clean_correct / self.clean_total

    @property
    def backdoor_accuracy(self) -> float:
        if self.backdoor_total == 0:
            return 0
        return self.backdoor_correct / self.backdoor_total

    @property
    def backdoored_accuracy(self) -> float:
        if self.backdoored_total == 0:
            return 0
        return self.backdoored_correct / self.backdoored_total

    def reset(self):
        for field in fields(self):
            setattr(self, field.name, field.default)

    def to_dict(self) -> dict:
        """Serialize dataclass including computed properties."""
        data = asdict(self)  # Serialize normal attributes
        props = {name: getattr(self, name) for name in dir(self.__class__)
                 if isinstance(getattr(self.__class__, name, None), property)}
        return {**data, **props}  # Merge both dictionaries
