NAME          toy05_lp_bounds
OBJSENSE
  MIN
ROWS
 N  OBJ
 G  ROW1
COLUMNS
    X1        OBJ               1.0   ROW1              1.0
    X2        OBJ              -2.0   ROW1              1.0
    X3        OBJ               3.0   ROW1              1.0
RHS
    RHS1      ROW1              1.0
BOUNDS
 FX BND1      X1                2.0
 LO BND1      X2                1.0
 UP BND1      X2                5.0
 LO BND1      X3               -2.0
 UP BND1      X3                4.0
ENDATA
