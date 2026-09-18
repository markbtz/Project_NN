"""
Utility condivise per il monitoraggio del training.
"""
import csv
import os

class EarlyStopping:
    """
    Tiene traccia delle epoche consecutive senza miglioramento.

    La decisione su cosa significhi "miglioramento" resta esterna:
    il training usa la stessa logica del best checkpoint.
    """

    def __init__(
        self,
        enabled: bool,
        patience: int,
    ):
        if patience < 1:
            raise ValueError(
                "early stopping patience deve essere >= 1."
            )

        self.enabled = enabled
        self.patience = patience
        self.epochs_without_improvement = 0

    def step(
        self,
        improved: bool,
    ) -> bool:
        """
        Aggiorna il contatore.

        Ritorna True quando il training deve fermarsi.
        """

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return (
            self.enabled
            and self.epochs_without_improvement
            >= self.patience
        )

class TrainingHistory:
    """
    Memorizza le statistiche del training epoca per epoca.
    """

    def __init__(self):
        self.records = []

    def add_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        selection_metric: str,
        selection_value: float,
    ):
        self.records.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
            }
        )

    def save_csv(
        self,
        path,
    ):
        """
        Salva la training history in formato CSV.
        """

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        fieldnames = [
            "epoch",
            "train_loss",
            "validation_loss",
            "selection_metric",
            "selection_value",
        ]

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                self.records
            )

    def load_csv(
        self,
        path,
    ):

        """ Carica una training history esistente, se presente.  """

        if not os.path.exists(path):
            return

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            self.records = [
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "validation_loss": float(
                        row["validation_loss"]
                    ),
                    "selection_metric": row[
                        "selection_metric"
                    ],
                    "selection_value": float(
                        row["selection_value"]
                    ),
                }
                for row in reader
            ]

    def save_loss_plot(
        self,
        path,
    ):
        """ Salva il grafico train loss / validation loss.  """

        if not self.records:
            return

        import matplotlib.pyplot as plt

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        epochs = [
            record["epoch"]
            for record in self.records
        ]

        train_losses = [
            record["train_loss"]
            for record in self.records
        ]

        validation_losses = [
            record["validation_loss"]
            for record in self.records
        ]

        plt.figure()

        plt.plot(
            epochs,
            train_losses,
            marker="o",
            label="Train loss",
        )

        plt.plot(
            epochs,
            validation_losses,
            marker="o",
            label="Validation loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            path,
            dpi=150,
        )
        plt.close()

    def consecutive_non_improving_epochs(
        self,
        selection_mode: str,
    ) -> int:
        """   Conta le epoche consecutive senza miglioramento alla fine della training history.
        """

        if not self.records:
            return 0

        if selection_mode not in (
            "max",
            "min",
        ):
            raise ValueError(
                "selection_mode deve essere 'max' oppure 'min'."
            )

        best_score = None
        epochs_without_improvement = 0

        for record in self.records:
            current_score = record[
                "selection_value"
            ]

            if best_score is None:
                best_score = current_score
                epochs_without_improvement = 0
                continue

            if selection_mode == "max":
                improved = current_score > best_score
            else:
                improved = current_score < best_score

            if improved:
                best_score = current_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        return epochs_without_improvement