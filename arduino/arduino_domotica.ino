/*
// =========================================================
// PANEL DE DOMÓTICA CONTROLADA POR VOZ
// Arduino UNO R3
//
// Comandos recibidos por Serial:
// LUZ_ON
// LUZ_OFF
// AIRE_ON
// AIRE_OFF
// BOMBA_REGAR
// BOMBA_OFF
// TODO_OFF
// TEST_AIRE
// TEST_BOMBA
// =========================================================


// =========================================================
// PINES
// =========================================================

const int LEDS[] = {2, 3, 4, 5, 6, 7};
const int NUM_LEDS = 6;

// Según la prueba anterior:
// BOMBA_REGAR activaba ventiladores cuando bomba estaba en D9.
// Por eso dejamos:
// AIRE  → D9
// BOMBA → D8
//
// Si en tu montaje real está al revés, intercambia estos valores.

const int PIN_RELE_AIRE = 9;
const int PIN_RELE_BOMBA = 8;


// =========================================================
// CONFIGURACIÓN DE RELÉS
// =========================================================

// Muchos módulos relé son activos en LOW:
// LOW  = encendido
// HIGH = apagado
//
// Si tu relé funciona al revés, cambia a:
// const int RELE_ON = HIGH;
// const int RELE_OFF = LOW;

const int RELE_ON = LOW;
const int RELE_OFF = HIGH;


// =========================================================
// TIEMPOS
// =========================================================

const unsigned long TIEMPO_RIEGO_MS = 3000;
const unsigned long TIEMPO_TEST_MS = 3000;


// =========================================================
// VARIABLES
// =========================================================

String comando = "";

bool bombaActiva = false;
unsigned long tiempoInicioBomba = 0;


// =========================================================
// SETUP
// =========================================================

void setup() {
  Serial.begin(9600);

  // Configurar LEDs
  for (int i = 0; i < NUM_LEDS; i++) {
    pinMode(LEDS[i], OUTPUT);
    digitalWrite(LEDS[i], LOW);
  }

  // Configurar relés
  pinMode(PIN_RELE_AIRE, OUTPUT);
  pinMode(PIN_RELE_BOMBA, OUTPUT);

  // Estado inicial seguro
  apagarAire();
  apagarBomba();
  apagarLuces();

  Serial.println("Arduino listo para recibir comandos.");
  Serial.print("PIN_RELE_AIRE = D");
  Serial.println(PIN_RELE_AIRE);
  Serial.print("PIN_RELE_BOMBA = D");
  Serial.println(PIN_RELE_BOMBA);
  Serial.println("Reles configurados como activos en LOW.");
}


// =========================================================
// LOOP PRINCIPAL
// =========================================================

void loop() {
  leerComandoSerial();
  controlarTiempoBomba();
}


// =========================================================
// LECTURA SERIAL
// =========================================================

void leerComandoSerial() {
  if (Serial.available() > 0) {
    comando = Serial.readStringUntil('\n');
    comando.trim();
    comando.toUpperCase();

    Serial.print("Comando recibido: ");
    Serial.println(comando);

    ejecutarComando(comando);
  }
}


// =========================================================
// CONTROL AUTOMÁTICO DE BOMBA
// =========================================================

void controlarTiempoBomba() {
  if (bombaActiva) {
    unsigned long tiempoActual = millis();

    if (tiempoActual - tiempoInicioBomba >= TIEMPO_RIEGO_MS) {
      apagarBomba();
      Serial.println("ACK_BOMBA_AUTO_OFF");
      Serial.println("Riego finalizado automaticamente.");
    }
  }
}


// =========================================================
// LUCES
// =========================================================

void encenderLuces() {
  for (int i = 0; i < NUM_LEDS; i++) {
    digitalWrite(LEDS[i], HIGH);
  }
}

void apagarLuces() {
  for (int i = 0; i < NUM_LEDS; i++) {
    digitalWrite(LEDS[i], LOW);
  }
}


// =========================================================
// AIRE
// =========================================================

void encenderAire() {
  digitalWrite(PIN_RELE_AIRE, RELE_ON);
}

void apagarAire() {
  digitalWrite(PIN_RELE_AIRE, RELE_OFF);
}


// =========================================================
// BOMBA
// =========================================================

void iniciarRiego() {
  if (!bombaActiva) {
    bombaActiva = true;
    tiempoInicioBomba = millis();

    digitalWrite(PIN_RELE_BOMBA, RELE_ON);

    Serial.println("Bomba encendida. Regando plantas...");
  } else {
    Serial.println("La bomba ya estaba activa. No se reinicia.");
  }
}

void apagarBomba() {
  digitalWrite(PIN_RELE_BOMBA, RELE_OFF);
  bombaActiva = false;
}


// =========================================================
// APAGADO GENERAL
// =========================================================

void apagarTodo() {
  apagarLuces();
  apagarAire();
  apagarBomba();
}


// =========================================================
// PRUEBAS DIRECTAS
// =========================================================

void testAire() {
  Serial.println("Probando rele de AIRE...");
  digitalWrite(PIN_RELE_AIRE, RELE_ON);
  delay(TIEMPO_TEST_MS);
  digitalWrite(PIN_RELE_AIRE, RELE_OFF);
  Serial.println("Test de AIRE finalizado.");
}

void testBomba() {
  Serial.println("Probando rele de BOMBA...");
  digitalWrite(PIN_RELE_BOMBA, RELE_ON);
  delay(TIEMPO_TEST_MS);
  digitalWrite(PIN_RELE_BOMBA, RELE_OFF);
  Serial.println("Test de BOMBA finalizado.");
}


// =========================================================
// EJECUCIÓN DE COMANDOS
// =========================================================

void ejecutarComando(String cmd) {
  if (cmd == "LUZ_ON") {
    encenderLuces();
    Serial.println("ACK_LUZ_ON");
    Serial.println("Luces encendidas.");
  }

  else if (cmd == "LUZ_OFF") {
    apagarLuces();
    Serial.println("ACK_LUZ_OFF");
    Serial.println("Luces apagadas.");
  }

  else if (cmd == "AIRE_ON") {
    encenderAire();
    Serial.println("ACK_AIRE_ON");
    Serial.println("Aire encendido.");
  }

  else if (cmd == "AIRE_OFF") {
    apagarAire();
    Serial.println("ACK_AIRE_OFF");
    Serial.println("Aire apagado.");
  }

  else if (cmd == "BOMBA_REGAR") {
    iniciarRiego();
    Serial.println("ACK_BOMBA_REGAR");
  }

  else if (cmd == "BOMBA_OFF") {
    apagarBomba();
    Serial.println("ACK_BOMBA_OFF");
    Serial.println("Bomba apagada.");
  }

  else if (cmd == "TODO_OFF") {
    apagarTodo();
    Serial.println("ACK_TODO_OFF");
    Serial.println("Todo apagado.");
  }

  else if (cmd == "TEST_AIRE") {
    testAire();
    Serial.println("ACK_TEST_AIRE");
  }

  else if (cmd == "TEST_BOMBA") {
    testBomba();
    Serial.println("ACK_TEST_BOMBA");
  }

  else {
    Serial.println("ACK_UNKNOWN");
    Serial.println("Comando no reconocido.");
    Serial.println("Comandos validos:");
    Serial.println("LUZ_ON, LUZ_OFF, AIRE_ON, AIRE_OFF, BOMBA_REGAR, BOMBA_OFF, TODO_OFF, TEST_AIRE, TEST_BOMBA");
  }
}

*/