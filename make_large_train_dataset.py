import pandas as pd

try:
    first_half = pd.read_csv('./data/bbc_news_train_3_first_half.csv')
    second_half = pd.read_csv('./data/bbc_news_train_3_second_half.csv')

    # Concatenate the data
    train_data = pd.concat([first_half, second_half], ignore_index=True)

    # Save the data
    train_data.to_csv('./data/bbc_news_train_3.csv', index=False)
    print('Data concatenated and saved successfully')
except Exception as e:
    print(f'Error: {e}')

