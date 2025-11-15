from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from extract import DataExtractor
from mapping import SessionTransitionProbMatrixTransformer, render_graph


if __name__ == "__main__":
    steps = [
        ('data_extraction', DataExtractor()),
        ('transition_matrix', SessionTransitionProbMatrixTransformer(threshold=0.05)),
    ]
    pipeline = Pipeline(steps)
    result = pipeline.fit_transform(None)
    print(f"Number of sessions: {len(result)}\n")

    for session_id, sess_data in result.items():
        fname = f"session_{session_id}"
        render_graph(fname, sess_data['matrix'], ls_index=sess_data['labels'], threshold=0.05, fmt="svg", view=False)
        print(f"Rendered {fname}.svg")
