// Streaming cell-averaging CFAR detector.
//
// Takes one power sample per clock (for example |X|^2 along one range line of
// the range-Doppler map) and decides, for every cell whose whole window fits
// in the line, whether it stands above alpha times the mean of its training
// cells:
//
//     |<- TRAIN ->|<- GUARD ->| CUT |<- GUARD ->|<- TRAIN ->|
//
// Two running sums (leading and lagging training cells) are updated with one
// add and one subtract each per sample, so the cost does not grow with the
// window. TRAIN is a power of two per side, so the mean is a shift, and alpha
// is fixed point with ALPHA_FRAC fractional bits.
//
// Bit exact with mira/hdl_model.py, checked by tests/test_hdl.py.
//
// Latency: 2 clocks after the newest cell of the window. Cells nearer than
// TRAIN + GUARD to either end of a line produce no output.

`timescale 1ns/1ps
`default_nettype none

module cfar_ca #(
    parameter integer DATA_W     = 16,  // unsigned power samples
    parameter integer GUARD      = 2,   // guard cells per side
    parameter integer TRAIN_LOG2 = 3,   // 2**TRAIN_LOG2 training cells per side
    parameter integer ALPHA_W    = 8,   // threshold factor width
    parameter integer ALPHA_FRAC = 4,   // fractional bits of alpha (Q4.4 by default)
    parameter integer INDEX_W    = 12   // supports lines up to 2**INDEX_W cells
) (
    input  wire                        clk,
    input  wire                        rst,
    // input stream
    input  wire                        in_valid,
    input  wire [DATA_W-1:0]           in_data,
    input  wire                        in_last,    // last cell of a line
    input  wire [ALPHA_W-1:0]          alpha,
    // one result per cell with a full window
    output reg                         out_valid,
    output reg  [INDEX_W-1:0]          out_index,  // position of the cell under test in its line
    output reg  [DATA_W-1:0]           out_data,
    output reg  [DATA_W+ALPHA_W-1:0]   out_threshold,
    output reg                         out_detect,
    output reg                         out_last    // last result of a line
);

    localparam integer TRAIN = 1 << TRAIN_LOG2;
    localparam integer C     = TRAIN + GUARD;        // cell under test, counted from the newest
    localparam integer L     = 2 * C + 1;            // window length
    localparam integer SUM_W = DATA_W + TRAIN_LOG2 + 1;

    // the window as one flat shift register, cell 0 (newest) in the low bits
    reg [DATA_W*L-1:0] win;
    `define CELL(n) win[DATA_W*(n) +: DATA_W]
    reg [SUM_W-1:0]   sum_lead, sum_lag;
    reg [INDEX_W-1:0] k;                             // index in its line of the newest sample
    reg               line_start;

    reg               s1_valid, s1_last;
    reg [INDEX_W-1:0] s1_k;

    wire [INDEX_W-1:0] k_next = line_start ? {INDEX_W{1'b0}} : k + 1'b1;

    always @(posedge clk) begin
        if (rst) begin
            win        <= {(DATA_W*L){1'b0}};
            sum_lead   <= {SUM_W{1'b0}};
            sum_lag    <= {SUM_W{1'b0}};
            k          <= {INDEX_W{1'b0}};
            line_start <= 1'b1;
            s1_valid   <= 1'b0;
            s1_last    <= 1'b0;
            s1_k       <= {INDEX_W{1'b0}};
        end else begin
            s1_valid <= 1'b0;
            if (in_valid) begin
                win <= {win[DATA_W*(L-1)-1:0], in_data};
                // a sample enters the leading training cells, the oldest one moves into the guard cells
                sum_lead <= sum_lead + in_data - `CELL(TRAIN-1);
                // a sample leaves the guard cells into the lagging training cells, the oldest drops out
                sum_lag  <= sum_lag + `CELL(TRAIN + 2*GUARD) - `CELL(L-1);
                k          <= k_next;
                line_start <= in_last;
                s1_valid   <= (k_next >= L - 1);         // whole window inside this line
                s1_last    <= in_last;
                s1_k       <= k_next;
            end
        end
    end

    // stage 2: threshold and compare on the window just shifted in
    wire [SUM_W-1:0]          total = sum_lead + sum_lag;
    wire [DATA_W-1:0]         noise = total >> (TRAIN_LOG2 + 1);  // mean of 2*TRAIN cells
    wire [DATA_W+ALPHA_W-1:0] product = noise * alpha;
    wire [DATA_W+ALPHA_W-1:0] thr = product >> ALPHA_FRAC;

    always @(posedge clk) begin
        if (rst) begin
            out_valid     <= 1'b0;
            out_index     <= {INDEX_W{1'b0}};
            out_data      <= {DATA_W{1'b0}};
            out_threshold <= {(DATA_W+ALPHA_W){1'b0}};
            out_detect    <= 1'b0;
            out_last      <= 1'b0;
        end else begin
            out_valid <= s1_valid;
            if (s1_valid) begin
                out_index     <= s1_k - C;
                out_data      <= `CELL(C);
                out_threshold <= thr;
                out_detect    <= ({{ALPHA_W{1'b0}}, `CELL(C)} > thr);
                out_last      <= s1_last;
            end
        end
    end

    `undef CELL

endmodule

`default_nettype wire
