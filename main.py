import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from src.model.training import train_rnn_model, RNNModel
from src.data.dataLoad import get_rates_between_dates
from src.logs.loggingProject import logger


def load_model(model_path, input_size, hidden_layer_size, output_size, num_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(
        input_size=input_size,
        hidden_size=hidden_layer_size,
        output_size=output_size,
        num_layers=num_layers
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()
    return model


def predict_next_rate(model, data, seq_len):
    """Использует модель для предсказания следующего значения."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_seq = data[-seq_len:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.item()


def main():
    logger.info("Запуск приложения")
    logger.debug(f"Текущий рабочий каталог: {os.getcwd()}")

    data_dir = os.path.join(os.getcwd(), 'src', 'data')

    config_path = os.path.join(os.getcwd(), "config/model.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Конфигурационный файл успешно загружен по пути: {config_path}")
    except Exception as e:
        logger.critical(f"ОШИБКА: {e}")
        exit(1)

    try:
        csv_filename = os.path.join(data_dir, f'data_{datetime.now().date()}.csv')
        if os.path.exists(csv_filename):
            existing_data = pd.read_csv(csv_filename)
            last_date = existing_data['date'].max()
            logger.info(f'Старые данные существуют. Последняя дата: {last_date}')
            start_date = datetime.fromisoformat(last_date)
        else:
            logger.info('Старые данные отсутствуют. Начинаем загрузку с начальной даты')
            existing_data = pd.DataFrame()
            start_date = datetime.fromisoformat(config['data']['start_date'])

        new_data = get_rates_between_dates(start_date, datetime.now(), config['data']['currencies'])

        if not existing_data.empty:
            new_data = new_data[~new_data['date'].isin(existing_data['date'])]

        if not new_data.empty:
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data = new_data.sort_values(by='date')
            new_data['rate_USD'] = new_data['rate_USD'].astype(np.float32)

            if os.path.exists(csv_filename):
                new_data.to_csv(csv_filename, mode='a', header=False, index=False)
            else:
                new_data.to_csv(csv_filename, index=False)

            logger.info(f'Данные успешно добавлены в файл: {csv_filename}')
        else:
            logger.info("Данные актуальны. Новые строки не добавлены")

        model_dir = os.path.join(os.getcwd(), 'src', 'model')
        models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

        if models:
            models.sort()
            model_path = os.path.join(model_dir, models[-1])
            logger.info(f"Загружаем модель: {model_path}")
            model = load_model(
                model_path,
                config['model']['input_size'],
                config['model']['hidden_layer_size'],
                config['model']['output_size'],
                config['model']['num_layers']
            )

            seq_len = config['data_prep']['seq_len']
            rates = existing_data['rate_USD'].values if not existing_data.empty else new_data['rate_USD'].values
            if len(rates) >= seq_len:
                next_rate = predict_next_rate(model, rates, seq_len)
                logger.info(f"Предсказанное значение курса: {next_rate}")

                next_date = datetime.now().date() + timedelta(days=1)
                prediction_row = pd.DataFrame({'date': [next_date], 'rate_USD': [next_rate]})
                prediction_row.to_csv(csv_filename, mode='a', header=False, index=False)
                logger.info(f"Предсказание добавлено в файл: {csv_filename}")
            else:
                logger.warning("Недостаточно данных для предсказания. Требуется больше данных.")
        else:
            logger.info('Модели отсутствуют, начинаем процесс обучения')
            train_rnn_model(
                data=new_data['rate_USD'].values if not new_data.empty else [],
                input_size=config['model']['input_size'],
                hidden_layer_size=config['model']['hidden_layer_size'],
                output_size=config['model']['output_size'],
                num_layers=config['model']['num_layers'],
                lr=config['train']['lr'],
                num_epochs=config['train']['num_epoch'],
                seq_len=config['data_prep']['seq_len'],
                model_name=config['model']['name']
            )

    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    main()
