import cv2
import numpy as np
import datetime
import threading
import sounddevice as sd
import speech_recognition as sr
import pyttsx3
import ollama

# =====================================================================
# CONFIGURAÇÕES DOS MODELOS E ÁUDIO
# =====================================================================
MODELO_TEXTO = "llama3"      # Respostas rápidas e conversas
MODELO_VISAO = "moondream"   # Usado APENAS quando você pedir para ele olhar
SAMPLE_RATE = 16000          # Taxa de amostragem padrão

# Variáveis Globais
ouvindo = False
processando = False
frame_atual = None

# Sintetizador de Voz (Text-to-Speech)
def falar(texto):
    def _falar_thread():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            voices = engine.getProperty('voices')
            for voice in voices:
                if any(p in voice.name.lower() for p in ["portuguese", "brazil", "maria", "daniel"]):
                    engine.setProperty('voice', voice.id)
                    break
            engine.say(texto)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERRO VOZ]: {e}")
    threading.Thread(target=_falar_thread, daemon=True).start()

# =====================================================================
# LÓGICA DE PROCESSAMENTO DO JARVIS
# =====================================================================

def processar_comando_jarvis(comando):
    global frame_atual, processando
    processando = True
    comando_lower = comando.lower()
    
    # Gatilhos para acionar a câmera
    gatilhos_visao = ["olha", "vendo", "o que e isso", "analise isso", "esta na minha mao", "foto"]
    precisa_visao = any(g in comando_lower for g in gatilhos_visao)

    try:
        # VISÃO SOB DEMANDA (Moondream)
        if precisa_visao and frame_atual is not None:
            falar("Analisando sensor óptico, chefe.")
            frame_pequeno = cv2.resize(frame_atual, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame_pequeno)
            img_bytes = buffer.tobytes()

            prompt = (
                "Você é o JARVIS. Analise a imagem e responda em português de forma concisa "
                f"em no máximo 2 frases. Pergunta: {comando}"
            )

            response = ollama.chat(
                model=MODELO_VISAO,
                messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}]
            )
        
        # CONVERSA DIRETA (Llama 3)
        else:
            prompt = (
                "Você é o JARVIS, a IA do capacete do Homem de Ferro. Responda em português "
                f"de forma direta, elegante e em no máximo 2 frases. Pergunta: {comando}"
            )
            response = ollama.chat(
                model=MODELO_TEXTO,
                messages=[{'role': 'user', 'content': prompt}]
            )

        texto_resposta = response['message']['content'].strip()
        print(f"\n[JARVIS]: {texto_resposta}")
        falar(texto_resposta)

    except Exception as e:
        print(f"[ERRO JARVIS]: {e}")
        falar("Houve uma falha no processamento dos dados.")
    
    processando = False

# =====================================================================
# CAPTURA DE ÁUDIO VIA SOUNDDEVICE (SEM PYAUDIO)
# =====================================================================

def escutar_microfone():
    global ouvindo
    recognizer = sr.Recognizer()

    print("[SISTEMA] Microfone ativo via SoundDevice...")

    while True:
        try:
            # 1. Escuta o ambiente em blocos de 3.5 segundos procurando pela palavra 'Jarvis'
            audio_data = sd.rec(int(3.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            
            byte_data = audio_data.tobytes()
            audio_sr = sr.AudioData(byte_data, SAMPLE_RATE, 2)
            
            texto = recognizer.recognize_google(audio_sr, language="pt-BR")
            print(f"[OUVIDO]: {texto}")

            if "jarvis" in texto.lower():
                ouvindo = True
                falar("Sim, chefe?")
                
                # 2. Ouve o comando de voz em um bloco mais longo (5.5 segundos)
                audio_cmd = sd.rec(int(5.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
                sd.wait()
                
                byte_cmd = audio_cmd.tobytes()
                audio_cmd_sr = sr.AudioData(byte_cmd, SAMPLE_RATE, 2)
                
                comando = recognizer.recognize_google(audio_cmd_sr, language="pt-BR")
                print(f"[COMANDO REGISTRADO]: {comando}")
                
                ouvindo = False
                
                # Envia o comando para a thread de processamento
                threading.Thread(target=processar_comando_jarvis, args=(comando,), daemon=True).start()

        except sr.UnknownValueError:
            ouvindo = False
        except Exception as e:
            ouvindo = False

# =====================================================================
# INTERFACE HUD COMPLETA (OPENCV)
# =====================================================================

def rodar_hud():
    global frame_atual
    cap = cv2.VideoCapture(0)
    
    CYAN = (255, 255, 0)
    AMARELO = (0, 255, 255)
    VERDE = (0, 255, 0)
    VERMELHO = (0, 0, 255)

    threading.Thread(target=escutar_microfone, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.flip(frame, 1)

        frame_atual = frame.copy()

        # Retículo Central
        cx, cy = 1280 // 2, 720 // 2
        cv2.circle(frame, (cx, cy), 45, CYAN, 1)
        cv2.line(frame, (cx - 65, cy), (cx + 65, cy), CYAN, 1)
        cv2.line(frame, (cx, cy - 65), (cx, cy + 65), CYAN, 1)

        # Barra Superior de Status
        agora = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.rectangle(frame, (30, 20), (1250, 60), (20, 20, 20), -1)
        cv2.rectangle(frame, (30, 20), (1250, 60), CYAN, 1)
        
        cv2.putText(frame, f"JARVIS OS | {agora}", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        
        if processando:
            status_str = "PROCESSANDO..."
            cor_status = AMARELO
        elif ouvindo:
            status_str = "ESCUTANDO COMANDO..."
            cor_status = VERMELHO
        else:
            status_str = "AGUARDANDO 'JARVIS'"
            cor_status = VERDE

        cv2.putText(frame, f"SISTEMA: {status_str}", (480, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 1)
        cv2.putText(frame, "BAT: 98% [|||||]", (1060, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMARELO, 1)

        # Telemetria Lateral
        cv2.rectangle(frame, (30, 100), (260, 220), (20, 20, 20), -1)
        cv2.rectangle(frame, (30, 100), (260, 220), CYAN, 1)
        cv2.putText(frame, "-- NAVEGACAO --", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMARELO, 1)
        cv2.putText(frame, "GPS: FIXADO", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, VERDE, 1)
        cv2.putText(frame, "ALT: 760m", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

        cv2.rectangle(frame, (1020, 100), (1250, 220), (20, 20, 20), -1)
        cv2.rectangle(frame, (1020, 100), (1250, 220), CYAN, 1)
        cv2.putText(frame, "-- TELEMETRIA --", (1030, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMARELO, 1)
        cv2.putText(frame, "REATOR: ESTAVEL", (1030, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, VERDE, 1)
        cv2.putText(frame, "TEMP INT: 24 C", (1030, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

        cv2.imshow("JARVIS HUD - Capacete System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rodar_hud()