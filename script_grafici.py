import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def creare_grafici(file_csv):
    """
    Legge il file CSV del benchmark e genera 4 grafici.
    """

    # --- 1. Caricamento e Controllo Dati ---
    try:
        df = pd.read_csv(file_csv)
    except FileNotFoundError:
        print(f"Errore: File non trovato: '{file_csv}'")
        print("Assicurati che 'benchmark_results.csv' sia nella stessa cartella dello script.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Errore: Il file '{file_csv}' è vuoto.")
        sys.exit(1)

    if df.empty:
        print("Il file CSV è vuoto. Nessun dato da plottare.")
        return

    print(f"Dati caricati con successo da '{file_csv}':")
    print(df.head())
    print("\nCreazione grafici in corso...")

    # Imposta uno stile grafico più pulito
    sns.set_theme(style="whitegrid")

    # --- 2. Grafico 1: Scalabilità del Tempo (Linee) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['NumImages'], df['CPUTimeMs'], marker='o', linestyle='--', color='tab:blue', label='CPU')
    plt.plot(df['NumImages'], df['CudaTimeMs'], marker='s', linestyle='-', color='lightgreen', label='CUDA')
    plt.xlabel('Numero di Immagini elaborate')
    plt.ylabel('Tempo Medio (ms) in Scala Logaritmica')
    plt.title('Scalabilità Tempo di Esecuzione (CPU vs CUDA)')
    plt.yscale('log')  # Scala logaritmica per vedere meglio la differenza
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xticks(df['NumImages']) # Assicura che i tick siano sui numeri di immagini testati

    file_salvataggio = 'grafico_scalabilita_tempo.png'
    plt.savefig(file_salvataggio)
    print(f"Grafico salvato: {file_salvataggio}")
    plt.close()

    # --- 3. Grafico 2: Scalabilità del Throughput (Linee) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['NumImages'], df['CPUThroughputMPS'], marker='o', linestyle='--', color='tab:blue', label='CPU')
    plt.plot(df['NumImages'], df['CudaThroughputMPS'], marker='s', linestyle='-', color='lightgreen', label='CUDA')
    plt.xlabel('Numero di Immagini elaborate')
    plt.ylabel('Throughput (MP/s)')
    plt.title('Scalabilità del Throughput (CPU vs CUDA)')
    plt.legend()
    plt.grid(True)
    plt.xticks(df['NumImages'])

    file_salvataggio = 'grafico_scalabilita_throughput.png'
    plt.savefig(file_salvataggio)
    print(f"Grafico salvato: {file_salvataggio}")
    plt.close()

    # --- 4. Grafico 3: Scalabilità dello Speedup (Linee) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['NumImages'], df['Speedup'], marker='x', linestyle='-', color='tab:blue', label='Speedup (CPU/CUDA)')
    # Linea di riferimento a 1x (dove CPU e CUDA hanno lo stesso tempo)
    plt.axhline(y=1, color='r', linestyle=':', label='1x (Parità)')
    plt.xlabel('Numero di Immagini elaborate')
    plt.ylabel('Speedup (x)')
    plt.title('Speedup (CUDA vs CPU)')
    plt.legend()
    plt.grid(True)
    plt.xticks(df['NumImages'])

    file_salvataggio = 'grafico_scalabilita_speedup.png'
    plt.savefig(file_salvataggio)
    print(f"Grafico salvato: {file_salvataggio}")
    plt.close()

    # --- 5. Grafico 4: Confronto MAX Workload (Barre) ---
    # Estrae l'ultima riga (carico di lavoro più grande)
    last_run = df.iloc[-1]
    max_images = int(last_run['NumImages'])

    # Crea una figura con 2 sotto-grafici (uno per il tempo, uno per il throughput)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Confronto Tempo Medio e Throughput per {max_images} Immagini', fontsize=16)

    # Sotto-grafico 1: Tempo
    labels = ['CPU', 'CUDA']
    tempi = [last_run['CPUTimeMs'], last_run['CudaTimeMs']]
    colori = ['tab:blue', 'lightgreen']
    bars1 = ax1.bar(labels, tempi, color=colori)
    ax1.set_ylabel('Tempo Medio (ms)')
    ax1.set_title('Confronto Tempo Medio')
    ax1.bar_label(bars1, fmt='%.2f ms')

    # Sotto-grafico 2: Throughput
    throughputs = [last_run['CPUThroughputMPS'], last_run['CudaThroughputMPS']]
    bars2 = ax2.bar(labels, throughputs, color=colori)
    ax2.set_ylabel('Throughput (MP/s)')
    ax2.set_title('Confronto Throughput Medio')
    ax2.bar_label(bars2, fmt='%.2f MP/s')

    file_salvataggio = 'grafico_confronto_max_workload.png'
    plt.savefig(file_salvataggio)
    print(f"Grafico salvato: {file_salvataggio}")
    plt.close()

    print("\nCompletato! 4 file .png sono stati creati.")

# --- Esecuzione Script ---
if __name__ == "__main__":
    nome_file_csv = 'benchmark_results.csv'
    creare_grafici(nome_file_csv)