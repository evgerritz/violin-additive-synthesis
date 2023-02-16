(
var gen_ps = {
	arg freq;
	var arr = Array.fill(20, {|i| p = i+1; SinOsc.ar(p*freq, mul: 1/p + rrand(-0.25*1/p,0.25*1/p))});
	Mix.ar(arr)
};

gen_ps.play(args: [\freq, 440]);
)


(
~noteGen = {|freq=196, duration=1.0, amp=1.0, vibrato=0.5, pan=0.0|
	var g3_p_amps = [-32.4.dbamp, -9.5.dbamp, -20.8.dbamp, -20.3.dbamp, -19.2.dbamp, -20.9.dbamp, -26.1.dbamp, -27.2.dbamp, -23.9.dbamp, -22.1.dbamp, -31.1.dbamp, -34.1.dbamp, -28.2.dbamp, -31.6.dbamp, -37.8.dbamp, -35.9.dbamp, -30.5.dbamp, -30.9.dbamp, -31.2.dbamp, -53.6.dbamp, -53.2.dbamp, -50.2.dbamp, -46.8.dbamp, -54.2.dbamp, -49.3.dbamp, -51.0.dbamp, -72.1.dbamp, -56.0.dbamp, -52.9.dbamp, -56.9.dbamp];

	var violin = Array.fill(20, {
		|i| p = i+1; SinOsc.ar(p*freq,
			mul: g3_p_amps[i])
	});
	Pan2.ar(Mix.ar(violin), pos:pan)

};

~noteGen.play(args: [\pan, 0.0]);
)
