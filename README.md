# KeyEmotions
 
project/
├── data/
│   ├── raw/               # Datos crudos: archivos MIDI originales
│   ├── processed/         # Datos preprocesados (tokens, secuencias)
│   ├── augmented/         # Datos con data augmentation
│   ├── splits/            # Divisiones de datos (train, val, test)
│   └── plots/             # Representaciones
├── experiments/            # Resultados y configuraciones de experimentos
│   ├── exp_0001/           # Primer experimento
│   │   ├── weights/        # Pesos del modelo entrenado
│   │   ├── results/        # Salidas generadas (archivos MIDI, gráficos, etc.)
│   │   ├── logs/           # Registros del entrenamiento
│   │   └── config.yaml     # Configuración utilizada en este experimento
│   ├── exp_0002/           # Segundo experimento
│   └── ...
├── notebooks/             # Notebooks para experimentos y análisis
├── src/                   # Código fuente principal del proyecto
│   ├── __init__.py        # Indica que esta carpeta es un paquete Python
│   ├── preprocessing/     # Código para preprocesar datos
│   │   ├── midi_preprocessing.py         # Procesamiento de archivos MIDI
│   │   ├── metadata_preprocessing.py     # Procesamiento de metadatos
│   │   ├── run_preprocessing.py          # Pipeline de preprocesamiento completo
<!-- │   │   ├── tokenizer.py   # La tokenización iría dentro de midi_preprocessing-->
│   │   └── data_augmentation.py   # Métodos de data augmentation
│   ├── models/            # Definición de modelos
│   │   ├── vae.py         # Implementación del Transformer VAE
│   │   ├── embeddings.py  # Implementación de embeddings (notas, emociones)
│   │   └── utils.py       # Funciones auxiliares para el modelo
│   ├── training/          # Código relacionado con el entrenamiento
│   │   ├── train.py       # Script principal para entrenar el modelo
│   │   ├── losses.py      # Definición de funciones de pérdida
│   │   ├── evaluation.py  # Evaluación del modelo
│   │   └── callbacks.py   # Callbacks para registrar métricas o guardar modelos
│   ├── utils/              # Funciones auxiliares
│   │   ├── general_utils.py               # Utilidades generales
│   │   ├── midi_utils.py                  # Funciones específicas para archivos MIDI
│   │   └── data_utils.py                  # Funciones para manejo de datos
│   │   ├── config.py      # Manejo de configuraciones del proyecto
│   │   └── plotting.py    # Herramientas para visualización de resultados
├── tests/                  # Pruebas automáticas
│   ├── test_preprocessing.py              # Pruebas de preprocesamiento
│   ├── test_training.py                   # Pruebas de entrenamiento
│   └── ...
├── scripts/               # Scripts auxiliares para tareas específicas
│   ├── preprocess_data.py # Preprocesamiento y tokenización de datos
│   ├── generate_music.py  # Generación de música con el modelo entrenado
│   └── analyze_results.py # Análisis de resultados y métricas
├── requirements.txt       # Dependencias del proyecto
├── setup.py                # Script de instalación del proyecto
├── README.md              # Descripción del proyecto
└── .gitignore             # Archivos y carpetas a ignorar en el control de versiones


main           -> Estado estable del proyecto
dev            -> Desarrollo activo y pruebas
feature/*      -> Nuevas características
bugfix/*       -> Corrección de errores
experiment/*   -> Experimentos y pruebas de nuevas configuraciones
hotfix/*       -> Correcciones urgentes en producción
docs           -> Actualización de documentación
tests          -> Desarrollo de pruebas y validaciones