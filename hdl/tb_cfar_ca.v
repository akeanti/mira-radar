// Testbench: streams cells from a text file through cfar_ca, with random idle
// cycles between them, and writes every result to another text file.
//
//   input  lines:  <data hex> <last 0|1>
//   output lines:  <index> <data> <threshold> <detect> <last>
//
// Run through tests/test_hdl.py, or by hand:
//   iverilog -g2005 -o tb hdl/cfar_ca.v hdl/tb_cfar_ca.v && vvp tb +IN=in.txt +OUT=out.txt +ALPHA=40

`timescale 1ns/1ps
`default_nettype none

module tb_cfar_ca;
    parameter integer DATA_W     = 16;
    parameter integer GUARD      = 2;
    parameter integer TRAIN_LOG2 = 3;
    parameter integer ALPHA_W    = 8;
    parameter integer ALPHA_FRAC = 4;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg in_valid = 1'b0;
    reg [DATA_W-1:0] in_data = 0;
    reg in_last = 1'b0;
    reg [ALPHA_W-1:0] alpha;

    wire out_valid, out_detect, out_last;
    wire [11:0] out_index;
    wire [DATA_W-1:0] out_data;
    wire [DATA_W+ALPHA_W-1:0] out_threshold;

    cfar_ca #(
        .DATA_W(DATA_W), .GUARD(GUARD), .TRAIN_LOG2(TRAIN_LOG2), .ALPHA_W(ALPHA_W), .ALPHA_FRAC(ALPHA_FRAC)
    ) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_data(in_data), .in_last(in_last), .alpha(alpha),
        .out_valid(out_valid), .out_index(out_index), .out_data(out_data), .out_threshold(out_threshold),
        .out_detect(out_detect), .out_last(out_last)
    );

    always #5 clk = ~clk;  // 100 MHz

    integer fin, fout, r, a, gap_pct, seed, n_in, n_out;
    reg [31:0] d;
    reg [31:0] l;
    reg [8*256-1:0] in_name, out_name;

    always @(posedge clk) begin
        if (out_valid) begin
            $fwrite(fout, "%0d %0d %0d %0d %0d\n", out_index, out_data, out_threshold, out_detect, out_last);
            n_out = n_out + 1;
        end
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_name)) in_name = "in.txt";
        if (!$value$plusargs("OUT=%s", out_name)) out_name = "out.txt";
        if (!$value$plusargs("ALPHA=%d", a)) a = 40;
        if (!$value$plusargs("GAP=%d", gap_pct)) gap_pct = 20;
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        alpha = a;
        n_in = 0;
        n_out = 0;
        fin = $fopen(in_name, "r");
        fout = $fopen(out_name, "w");
        if (fin == 0) begin
            $display("cannot open %0s", in_name);
            $finish;
        end

        repeat (4) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);

        while (!$feof(fin)) begin
            r = $fscanf(fin, "%h %h\n", d, l);
            if (r == 2) begin
                // idle cycles between samples: the core must only move on in_valid
                while (($random(seed) % 100 + 100) % 100 < gap_pct) begin
                    in_valid <= 1'b0;
                    @(posedge clk);
                end
                in_valid <= 1'b1;
                in_data  <= d[DATA_W-1:0];
                in_last  <= l[0];
                n_in = n_in + 1;
                @(posedge clk);
            end
        end
        in_valid <= 1'b0;
        repeat (8) @(posedge clk);
        $fclose(fin);
        $fclose(fout);
        $display("cfar_ca: %0d cells in, %0d results out", n_in, n_out);
        $finish;
    end
endmodule

`default_nettype wire
