from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from extract import DataExtractor
from mapping import SessionTransitionProbMatrixTransformer, render_graph


if __name__ == "__main__":
    steps = [
        ('data_extraction', DataExtractor()),
        #('transition_matrix', SessionTransitionProbMatrixTransformer(threshold=0.05)),
    ]
    pipeline = Pipeline(steps)
    result = pipeline.fit_transform(None)
    print(result)
    print(result.info())
