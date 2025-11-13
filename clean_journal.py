import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
import argparse

def clean_journal(old : optuna.study.Study, new : optuna.study.Study) -> None:
	old_trials = old.get_trials(deepcopy=False)
	for trial in old_trials:
		if trial.state in (TrialState.RUNNING, TrialState.WAITING):
			continue
		else:
			new.add_trial(trial)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Clean Optuna journal file by removing RUNNING and WAITING trials.")
	parser.add_argument("--input", "-i", type=str, help="Path to the input journal file.")
	parser.add_argument("--output", "-o", type=str, help="Path to the output cleaned journal file.", default="cleaned_journal.log")
	parser.add_argument("--study_name","-s", type=str, help="Name of the study to clean.", default="PiPPINN_HPO")
	args = parser.parse_args()

	# Load the old journal storage
	old_storage = JournalStorage(JournalFileBackend(args.input))
	old_study = optuna.load_study(study_name=args.study_name, storage=old_storage)

	# Create a new journal storage
	new_storage = JournalStorage(JournalFileBackend(args.output))
	new_study = optuna.create_study(study_name=args.study_name, storage=new_storage)

	# Clean the journal
	clean_journal(old_study, new_study)

	print(f"Cleaned journal saved to {args.output}")
