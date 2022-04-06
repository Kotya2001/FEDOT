from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_classification_example(timeout: float = None):
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    # baseline_model = Fedot(problem=problem, timeout=timeout)
    # baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')
    #
    # baseline_model.predict(features=test_data_path)
    # print(baseline_model.get_metrics())zzzzzzzzzzzzz

    auto_model = Fedot(problem=probzlem, seed=42, timeout=timeout,
                       composer_params={'with_tuning': False, 'collect_mode_metric': True}, n_jobs=1)
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()
    auto_model.history.save('history.json')
    return prediction


if __name__ == '__main__':
    run_classification_example(timeout=2)
