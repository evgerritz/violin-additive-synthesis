(
var gen_ps = {
	arg freq;
	var arr = Array.fill(20, {|i| p = i+1; Env.adsr.kr(2) * SinOsc.ar(p*freq + LFNoise1(), mul: 1/p + rrand(-0.25*1/p,0.25*1/p))});
	Mix.ar(arr)
};

y = gen_ps.play(args: [\freq, 440]);
//y.set(\gate, 0)
)



(
{ SinOsc.ar(
        freq: LFNoise0.ar(freq: 1, mul: 400, add: 450),
        phase: 0, mul: 0.2
    )
}.play;
)



{ LFNoise1.ar(500, 100, 2) }.plot(0.25);