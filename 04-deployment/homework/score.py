import argparse
import pickle
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    return parser.parse_args()


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    categorical = ['PULocationID', 'DOLocationID']

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    args = parse_args()
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')

    return df


def prepare_dictionaries(df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return model, dv


def apply_model(input_file, output_file):
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    model, dv = load_model()

    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    mean_prediction_duration = np.mean(y_pred)

    print(f"Mean predicted duration: {mean_prediction_duration:.2f} minutes")

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    args = parse_args()
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'

    apply_model(
        input_file=input_file,
        output_file=output_file
    )


if __name__ == '__main__':
    run()
