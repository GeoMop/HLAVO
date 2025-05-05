
#define PIN_ON 47 // napajeni !!!

// const int windvane_pin = A0; 
// const int anemometer_pin = 5; 
const int raingauge_pin = 6; // 10 kOhm / 10pF


volatile int edge_count = 0;  // Counter for the falling edges, declared volatile

volatile uint32_t lastInterruptTime = 0;
const uint32_t debounceThreshold = 50; // Debounce threshold in milliseconds

void setup() {
  Serial.begin(115200);

//   // for version over 3.5 need to turn uSUP ON
  pinMode(PIN_ON, OUTPUT);      // Set EN pin for uSUP stabilisator as output
  digitalWrite(PIN_ON, HIGH);   // Turn on the uSUP power

  pinMode(raingauge_pin, INPUT_PULLUP);  // Set GPIO 4 as input with pull-up
  attachInterrupt(digitalPinToInterrupt(raingauge_pin), countFallingEdge, CHANGE);  // Attach interrupt for falling edge
}

void loop() {

  while (true) {
    Serial.print("Number of falling edges: ");
    Serial.println(edge_count);
    delay(1000);  // Update every second
  }
}

// ISR to increment the edge counter
// void countFallingEdge() {
//   edge_count++;
// }

void IRAM_ATTR countFallingEdge() {
    uint32_t currentTime = millis();
    if (currentTime - lastInterruptTime > debounceThreshold) {
        edge_count++;
        lastInterruptTime = currentTime;
    }
}
