import torch
from colorama import Fore, Style, init
init()  # Initialize colorama for Windows
from warnings import filterwarnings
filterwarnings('ignore')
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset.preprocessing import prepare_data
from model import train, test, fine_tune
import datetime
from vissualize import plot_confusion_matrix
import pickle
import constants



#set manuel seed
torch.manual_seed(42)

def main():
    start_time = time.time()

    torch.cuda.empty_cache()


    print(f"Loading the data...")
    # Load the data
    train_df = prepare_data("./data/bbc_news_train_2.csv", text='Text', augment=True, categories=True)
    test_df = prepare_data("./data/bbc_news_tests.csv", text='Text', augment=False)
    scraped_df = prepare_data("./data/scraped/bbc_articles.csv", text='body', augment=False, stratified_sampling=True)

    print(f"{Fore.YELLOW}Train data shape: {train_df.shape}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Scraped data shape: {scraped_df.shape}{Style.RESET_ALL}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #train the model
    print(f"{Fore.YELLOW}Training the model on {device}...{Style.RESET_ALL} ")
    best_loss, best_accuracy = train(train_df, device, constants.folds, constants.epochs)
    end_time = time.time()
    print(f"{Fore.GREEN}Best Validation Loss: {best_loss:.6f}, Best Validation Accuracy: {best_accuracy * 100:.2f}%{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Total time taken: {((end_time - start_time)/60):.2f} minutes{Style.RESET_ALL}")

    #test the model
    model_path = './results/best_model.pth'
    test_accuracy, test_predictions, test_labels = test(test_df, device, model_path, feature_col='Text', label_col='Category')
    print(f"{Fore.WHITE}Test Accuracy: {test_accuracy * 100:.2f}%{Style.RESET_ALL}")

    #test the model on the scraped data
    scraped_accuracy, scraped_predictions, scraped_labels = test(scraped_df, device, model_path, feature_col='body', label_col='Category')
    print(f"{Fore.WHITE}Scraped Data Accuracy: {scraped_accuracy * 100:.2f}%{Style.RESET_ALL}")

    #fine tune the model
    fine_tune(scraped_df, device, epochs=50, percent_to_train=0.5)

    #test the fine tuned model
    model_path = './results/fine_tuned_model.pth'
    tuned_accuracy, tuned_predictions, tuned_labels = test(scraped_df, device, model_path, feature_col='body', label_col='Category')
    print(f"{Fore.WHITE}Fine Tuned Test Accuracy: {tuned_accuracy * 100:.2f}%{Style.RESET_ALL}")

    #load the category to int mapping for plotting the confusion matrix
    with open('category_to_int.pkl', 'rb') as f:
        category_to_int = pickle.load(f)

    # Plot the confusion matrix
    classes = list(category_to_int.keys()) # Get the class names
    plot_confusion_matrix(test_labels, test_predictions, classes, title='Test Confusion Matrix', file_name='./results/test_confusion_matrix.png') 
    plot_confusion_matrix(scraped_labels, scraped_predictions, classes, title='Scraped Data Confusion Matrix', file_name='./results/scraped_confusion_matrix.png')
    plot_confusion_matrix(tuned_labels, tuned_predictions, classes, title='Fine Tuned Data Confusion Matrix', file_name='./results/fine_tuned_confusion_matrix.png')

    #Log the results to a file
    with open('./results/best_accuracy_Log.txt', 'a') as f:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        f.write(f"{current_date} - Best Validation Loss: {best_loss:.6f}, Best Validation Accuracy: {best_accuracy * 100:.2f}%, Test accuracy: {test_accuracy * 100:.2f}, Scraped accuracy {scraped_accuracy * 100:.2f}, Fine tunned accuracy {tuned_accuracy * 100:.2f}, Total time taken: {((end_time - start_time)/60):.2f} minutes\n")

if __name__ == "__main__":
    main()
