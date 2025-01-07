# KeyEmotions
 
project/
├── data/
│   ├── raw/               # Datos crudos: archivos MIDI originales
│   ├── processed/         # Datos preprocesados (tokens, secuencias)
│   ├── augmented/         # Datos con data augmentation
│   └── splits/            # Divisiones de datos (train, val, test)
├── notebooks/             # Notebooks para experimentos y análisis
├── src/                   # Código fuente principal del proyecto
│   ├── __init__.py        # Indica que esta carpeta es un paquete Python
│   ├── preprocessing/     # Código para preprocesar datos
│   │   ├── midi_loader.py # Carga y extracción de datos MIDI
│   │   ├── tokenizer.py   # Tokenización de secuencias
│   │   └── augmenter.py   # Métodos de data augmentation
│   ├── models/            # Definición de modelos
│   │   ├── vae.py         # Implementación del Transformer VAE
│   │   ├── embeddings.py  # Implementación de embeddings (notas, emociones)
│   │   └── utils.py       # Funciones auxiliares para el modelo
│   ├── training/          # Código relacionado con el entrenamiento
│   │   ├── train.py       # Script principal para entrenar el modelo
│   │   ├── losses.py      # Definición de funciones de pérdida
│   │   ├── evaluation.py  # Evaluación del modelo
│   │   └── callbacks.py   # Callbacks para registrar métricas o guardar modelos
│   ├── utils/             # Funciones y herramientas generales
│   │   ├── logger.py      # Configuración de registros (logs)
│   │   ├── config.py      # Manejo de configuraciones del proyecto
│   │   └── plotting.py    # Herramientas para visualización de resultados
├── experiments/           # Resultados y configuraciones de experimentos
│   ├── configs/           # Configuraciones YAML o JSON para distintos experimentos
│   ├── logs/              # Registros generados durante el entrenamiento
│   ├── checkpoints/       # Pesos del modelo guardados
│   └── samples/           # Piezas musicales generadas por el modelo
├── tests/                 # Pruebas unitarias para garantizar la calidad del código
├── scripts/               # Scripts auxiliares para tareas específicas
│   ├── preprocess_data.py # Preprocesamiento y tokenización de datos
│   ├── generate_music.py  # Generación de música con el modelo entrenado
│   └── analyze_results.py # Análisis de resultados y métricas
├── requirements.txt       # Dependencias del proyecto
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