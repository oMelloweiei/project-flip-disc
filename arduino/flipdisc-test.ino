#define DATA_PIN D1  // TPIC6B595 for Data pin
#define CLOCK_PIN D3 // TPIC6B595 for Clock pin
#define CLEAR_PIN D7 // TPIC6B595 for Clear pin

#define LATCH_ROW_PIN D2 // TPIC6B595 for ROW Latch pin
#define ROW_EN D4        // TPIC6B595 for ROW Output Enable pin (active LOW)

#define LATCH_COL_PIN D6 // TPIC6B595 for COLUMN Latch pin
#define COL_EN D8        // TPIC6B595 for COLUMN Output Enable pin (active LOW)

// this setup use 2 shift register with 2 darlington transistor array

// data = 0000000000000000
// it need to drive likes this = 0000000011111111 for do row_high col_low or row_low col_high

void setup()
{

    pinMode(DATA_PIN, OUTPUT);
    pinMode(CLOCK_PIN, OUTPUT);
    pinMode(CLEAR_PIN, OUTPUT);

    // ROW control pins
    pinMode(LATCH_ROW_PIN, OUTPUT);
    pinMode(ROW_EN, OUTPUT);

    // COLUMN control pins
    pinMode(LATCH_COL_PIN, OUTPUT);
    pinMode(COL_EN, OUTPUT);

    // Disable outputs initially
    digitalWrite(COL_EN, HIGH);
    digitalWrite(ROW_EN, HIGH);

    digitalWrite(CLEAR_PIN, LOW);
    digitalWrite(CLEAR_PIN, HIGH);

    // Enable ROW and COLUMN outputs
    digitalWrite(ROW_EN, LOW);
    digitalWrite(COL_EN, LOW);

    digitalWrite(LATCH_ROW_PIN, LOW);
    digitalWrite(LATCH_ROW_PIN, HIGH);
    digitalWrite(LATCH_COL_PIN, LOW);
    digitalWrite(LATCH_COL_PIN, HIGH);

    // Initialize serial communication
    Serial.begin(115200);
    Serial.println("Flip-disc Test Initialized");
}

// Turn off all rows
void allRowsOff()
{
    digitalWrite(LATCH_ROW_PIN, LOW);

    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b10101010);

    digitalWrite(LATCH_ROW_PIN, HIGH);
}

// Set all rows high
void allRowsHigh()
{
    digitalWrite(LATCH_ROW_PIN, LOW);

    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b00000000);

    digitalWrite(LATCH_ROW_PIN, HIGH);
}

// Set all rows low
void allRowsLow()
{
    digitalWrite(LATCH_ROW_PIN, LOW);

    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b11111111);
    digitalWrite(LATCH_ROW_PIN, HIGH);
}

// Write 0b10101010 (off pattern) into colVec
// void colVecOff() {

//     colVec[c] = 0b10101010;

// }

// Shift colVec out to columns
void shiftColVec()
{
    digitalWrite(LATCH_COL_PIN, LOW);
    digitalWrite(LATCH_COL_PIN, HIGH);
}

// Turn off all columns
void allColsOff()
{
    digitalWrite(LATCH_COL_PIN, LOW);

    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b10101010);

    digitalWrite(LATCH_COL_PIN, HIGH);
    // colVecOff();
    // shiftColVec();
}

// Set all columns low
void allColsLow()
{
    digitalWrite(LATCH_COL_PIN, LOW);
    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b11111111);
    digitalWrite(LATCH_COL_PIN, HIGH);
}

// Set all columns high
void allColsHigh()
{
    digitalWrite(LATCH_COL_PIN, LOW);

    shiftOut(DATA_PIN, CLOCK_PIN, LSBFIRST, 0b00000000);

    digitalWrite(LATCH_COL_PIN, HIGH);
}

void loop()
{
    allRowsLow();
    allColsHigh();
    Serial.println("Row Low , Col High");
    delay(1000);

    allRowsOff();
    allColsOff();
    Serial.println("Row Off , Col Off");
    delay(3000);

    allColsLow();
    allRowsHigh();
    Serial.println("Row High , Col Low");
    delay(1000);

    allRowsOff();
    allColsOff();
    Serial.println("Row Off , Col Off");
    delay(3000);
}
