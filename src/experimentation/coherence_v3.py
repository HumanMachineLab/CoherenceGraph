import config, os, sys, random, string, csv
import pandas as pd

from dataclasses import dataclass, asdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, config.root_path)


from db.dbv2 import Table, AugmentedTable, TrainTestTable
from src.dataset.utils import flatten, dedupe_list, truncate_string
from src.encoders.coherence_v3 import Coherence
from src.experimentation.prediction_thresholds import thresholds
from src.experimentation.graphs import display_pk_wd_proximity

from utils.metrics import windowdiff, pk, get_proximity


def get_random_hash(k):
    x = "".join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


experiment_set_hash = get_random_hash(
    5
)  # global hash for all the experiments during this run.


@dataclass
class CoherenceExperiment:
    num_samples: int = 50  # how many samples to test on
    start: int = 0  # where to start the predictions from in the dataset
    dataset_type: str = "city"  # either city or disease
    model_string: str = "bert-base-uncased"
    max_words_per_step: int = 4
    coherence_threshold: float = 0.5
    experiment_hash: str = None  # a unique identifier for the experiment
    batch_size: int = 1  # number of samples to pull keywords from at a time.
    max_graph_depth: int = 7 # the maximum depth for the coherence graph

    keyword_diversity: float = (
        0.0  # diversity value for mmr. the higher, the more diverse the keywords are.
    )

    diverse_keywords: bool = False
    similar_keywords: bool = True
    ablation: bool = False  # if set to True, remove the coherence map

    # debugging
    print_metrics_summary: bool = (False,)
    print_predictions_summary: bool = (False,)
    show_graphs: bool = True


class PredictByChainCountExperiment:
    def __init__(self):
        self.experiments = []

    def queue_experiment(self, experiment: CoherenceExperiment):
        experiment.experiment_hash = get_random_hash(5)
        self.experiments.append(experiment)

    def get_experiments(self):
        return self.experiments

    # gather all the predictions and store them in a csv.
    def log_predictions(
        self, experiment: CoherenceExperiment, predictions, true_labels
    ):
        predictions_directory = "{}/predictions/{}".format(
            config.root_path, experiment_set_hash
        )

        if not os.path.exists(predictions_directory):
            os.makedirs(predictions_directory)

        predictions_path = "{}/predictions.csv".format(predictions_directory)

        # don't add headers to the file if it already exists.
        hdr = False if os.path.isfile(predictions_path) else True

        # turn the experiment dataclass into a dictionary so we can put it into a csv file
        experiment_dict = asdict(experiment)

          # convert pytorch tensor to raw primitive type before storage
        experiment_dict["predictions"] = predictions
        experiment_dict["true_labels"] = true_labels

        df = pd.json_normalize(experiment_dict)

        # append data frame to CSV file
        df.to_csv(predictions_path, mode="a", index=False, header=hdr)

        self.log_evaluations(experiment, predictions, true_labels)

    # calculate all the evaluation metrics and store them in a csv file.
    def log_evaluations(self, experiment, predictions, true_labels):

        evaluation_directory = "{}/predictions/{}/{}".format(
            config.root_path, experiment_set_hash, experiment.experiment_hash
        )

        if not os.path.exists(evaluation_directory):
            os.makedirs(evaluation_directory)

        evaluation_file = "{}/eval.csv".format(evaluation_directory)

        # don't add headers to the file if it already exists.
        hdr = False if os.path.isfile(evaluation_file) else True

        df_data = []

        if experiment.print_metrics_summary:
            print()
            print("============= Metrics Summary =============")

        lowest_pk = 1
        lowest_pred_thresh = 1
        best_proximity = 0
        best_overall_score = 1
        best_predictions = None
        pks = []
        wds = []
        proximities = []

        avg_k = len(true_labels) // (
            true_labels.count(1) + 1
        )  # get avg segment size

        # convert to strings so it can be used with the wd and pk metrics
        pred_string = "".join(map(str,predictions))
        true_string = "".join(map(str,true_labels))

        # calculate all the metrics we will be storing

        print(pred_string)
        print(true_string)
        print(len(pred_string), len(true_string))
        wd_score = windowdiff(pred_string, true_string, avg_k)
        pk_score = pk(pred_string, true_string, avg_k)
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="micro"
        )
        proximity, _, _, _ = get_proximity(true_labels, predictions)

        pks.append(pk_score)
        wds.append(wd_score)
        proximities.append(proximity)

        # append all the data to an array before converting to a dataframe below
        df_data.append(
            [
                avg_k,
                tp,
                fp,
                tn,
                fn,
                precision,
                recall,
                f1,
                pk_score,
                wd_score,
                proximity,
            ]
        )
        # calculate lowest pred thresh and pk for summary printing
        if (pk_score - proximity) < best_overall_score:
            lowest_pk = pk_score
            best_predictions = predictions
            best_proximity = proximity
            best_overall_score = pk_score - proximity

        if experiment.print_metrics_summary:
            print("pk score:", pk_score)
            print("wd score:", wd_score)
            print("proximity:", proximity)
            print(
                f"confusion: f1 [{f1}], tp [{tp}], fp [{fp}], tn [{tn}], fn [{fn}]"
            )
            print("==========================")

        df_evaluation_set = pd.DataFrame(
            df_data,
            columns=[
                "K",
                "TP",
                "FP",
                "TN",
                "FN",
                "precision",
                "recall",
                "f1",
                "Pk",
                "WindowDiff",
                "Proximity",
            ],
        )

        # append data frame to CSV file
        df_evaluation_set.to_csv(evaluation_file, mode="a", index=False, header=hdr)

        # if experiment.show_graphs:
        #     display_pk_wd_proximity(curr_model_thresholds, pks, wds, proximities)

        if experiment.print_predictions_summary:
            print("============= Predictions Summary =============")
            print(
                f"best pk: {lowest_pk}, best prediction threshold: {lowest_pred_thresh}, proximity: {best_proximity}"
            )
            print(f"P:{best_predictions}")
            print(f"R:{true_labels}")

    # run the actual experiment
    def run(self):
        print(f"Running experiment set: {experiment_set_hash}")
        for i, experiment in enumerate(self.experiments):
            print(f"Running experiment: {experiment}")
            table = Table(experiment.dataset_type)

            all_segments = table.get_all_segments()

            segments = [[y[1] for y in x] for x in all_segments]
            segments_labels = [
                [1 if i == 0 else 0 for i, y in enumerate(x)] for x in all_segments
            ]

            flattened_segments = flatten(segments)
            flattened_labels = flatten(segments_labels)

            segments_to_test = flattened_segments[
                experiment.start : experiment.start + experiment.num_samples
            ]
            labels_to_test = flattened_labels[
                experiment.start : experiment.start + experiment.num_samples
            ]

            batch_segments_to_test = segments_to_test[
                0 : experiment.batch_size
                * (len(segments_to_test) // experiment.batch_size)
            ]
            batch_labels_to_test = labels_to_test[
                0 : experiment.batch_size
                * (len(labels_to_test) // experiment.batch_size)
            ]

            # initialize the coherence library
            coherence = Coherence(
                max_words_per_step=experiment.max_words_per_step,
                model_string=experiment.model_string,
                coherence_threshold=experiment.coherence_threshold,
                keyword_diversity=experiment.keyword_diversity,
                diverse_keywords=experiment.diverse_keywords,
                similar_keywords=experiment.similar_keywords,
                ablation=experiment.ablation,
            )

            logits = coherence.predict_by_chain_count(
                text_data=batch_segments_to_test,
                max_tokens=128,
                max_graph_depth=experiment.max_graph_depth,
                batch_size=experiment.batch_size,
            )

            self.log_predictions(experiment, logits, batch_labels_to_test)

            print(f"\nExperiment {i+1} - {experiment.experiment_hash} complete.")
            print("==============================================\n")
