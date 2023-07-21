import requests
import json
import os
import io
import config
import chess.pgn
from tqdm import tqdm
import concurrent.futures
import time


chess_com_gm_elos = (3411, 2950, 2882, 2882, 2835, 2828, 2814, 2812, 2800, 2790, 2785, 2785, 2780, 2780, 2780, 2776, 2773, 2770, 2768, 2768, 2765, 2764, 2753, 2751, 2742, 2740, 2733, 2731, 2730, 2729, 2729, 2728, 2728, 2726, 2722, 2721, 2718, 2717, 2715, 2714, 2711, 2709, 2706, 2705, 2703, 2703, 2701, 2701, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2699, 2699, 2699, 2698, 2697, 2695, 2691, 2690, 2690, 2689, 2689, 2688, 2686, 2686, 2686, 2685, 2685, 2684, 2683, 2682, 2680, 2680, 2679, 2678, 2677, 2677, 2676, 2675, 2675, 2675, 2675, 2673, 2673, 2671, 2670, 2668, 2667, 2665, 2664, 2664, 2663, 2663, 2662, 2660, 2660, 2660, 2658, 2657, 2657, 2655, 2655, 2654, 2654, 2653, 2652, 2651, 2651, 2651, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2649, 2649, 2647, 2646, 2646, 2645, 2644, 2643, 2643, 2642, 2641, 2640, 2640, 2640, 2639, 2637, 2636, 2635, 2635, 2634, 2634, 2633, 2631, 2630, 2630, 2630, 2629, 2629, 2627, 2627, 2627, 2627, 2626, 2626, 2625, 2624, 2624, 2624, 2624, 2623, 2622, 2621, 2621, 2620, 2620, 2620, 2620, 2619, 2618, 2618, 2618, 2617, 2616, 2616, 2616, 2616, 2615, 2613, 2612, 2611, 2611, 2611, 2610, 2609, 2609, 2609, 2609, 2609, 2608, 2607, 2605, 2604, 2604, 2604, 2604, 2604, 2603, 2603, 2603, 2603, 2602, 2601, 2601, 2601, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2600, 2599, 2599, 2599, 2598, 2597, 2597, 2597, 2596, 2595, 2595, 2595, 2594, 2594, 2593, 2593, 2593, 2592, 2592, 2592, 2592, 2591, 2591, 2591, 2591, 2590, 2590, 2589, 2589, 2589, 2589, 2589, 2588, 2588, 2588, 2588, 2587, 2587, 2586, 2585, 2585, 2584, 2583, 2583, 2582, 2582, 2582, 2581, 2581, 2580, 2580, 2580, 2579, 2578, 2578, 2578, 2577, 2577, 2577, 2577, 2577, 2576, 2575, 2575, 2575, 2575, 2575, 2574, 2574, 2573, 2572, 2572, 2572, 2571, 2571, 2570, 2570, 2570, 2570, 2569, 2569, 2569, 2567, 2567, 2567, 2566, 2566, 2565, 2564, 2563, 2563, 2562, 2562, 2562, 2561, 2560, 2560, 2560, 2560, 2560, 2559, 2558, 2558, 2557, 2556, 2556, 2556, 2556, 2555, 2555, 2555, 2555, 2554, 2554, 2553, 2553, 2553, 2552, 2552, 2551, 2551, 2551, 2551, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2550, 2549, 2549, 2549, 2549, 2548, 2548, 2548, 2547, 2546, 2546, 2546, 2546, 2546, 2545, 2545, 2545, 2544, 2543, 2543, 2542, 2542, 2541, 2541, 2540, 2540, 2540, 2540, 2538, 2538, 2538, 2537, 2537, 2537, 2537, 2537, 2536, 2536, 2536, 2535, 2535, 2535, 2534, 2534, 2534, 2533, 2532, 2532, 2532, 2532, 2531, 2531, 2531, 2531, 2531, 2530, 2530, 2530, 2529, 2529, 2528, 2528, 2528, 2528, 2528, 2527, 2527, 2527, 2527, 2527, 2526, 2526, 2526, 2526, 2526, 2525, 2525, 2525, 2525, 2525, 2525, 2525, 2524, 2524, 2524, 2524, 2523, 2523, 2523, 2523, 2522, 2522, 2522, 2521, 2521, 2521, 2521, 2520, 2520, 2520, 2519, 2518, 2518, 2518, 2518, 2517, 2517, 2517, 2517, 2516, 2516, 2515, 2515, 2514, 2514, 2514, 2514, 2513, 2513, 2513, 2513, 2512, 2512, 2512, 2512, 2511, 2511, 2511, 2511, 2511, 2511, 2511, 2510, 2510, 2510, 2510, 2509, 2509, 2508, 2508, 2507, 2507, 2507, 2507, 2506, 2506, 2505, 2504, 2504, 2504, 2504, 2503, 2503, 2502, 2502, 2502, 2502, 2502, 2501, 2501, 2501, 2501, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2499, 2499, 2498, 2498, 2498, 2498, 2497, 2497, 2497, 2496, 2496, 2496, 2496, 2496, 2495, 2495, 2495, 2495, 2494, 2494, 2494, 2494, 2494, 2493, 2493, 2493, 2492, 2492, 2491, 2491, 2491, 2490, 2490, 2490, 2490, 2490, 2489, 2489, 2489, 2489, 2488, 2488, 2487, 2486, 2485, 2485, 2484, 2484, 2484, 2483, 2482, 2480, 2480, 2480, 2479, 2478, 2478, 2476, 2476, 2475, 2474, 2473, 2473, 2473, 2472, 2472, 2472, 2471, 2470, 2469, 2468, 2465, 2465, 2465, 2465, 2465, 2465, 2464, 2463, 2463, 2462, 2461, 2461, 2460, 2460, 2460, 2456, 2456, 2455, 2454, 2454, 2454, 2453, 2452, 2451, 2451, 2451, 2451, 2450, 2450, 2450, 2450, 2449, 2447, 2447, 2445, 2444, 2443, 2441, 2440, 2439, 2438, 2438, 2436, 2436, 2435, 2434, 2432, 2431, 2431, 2431, 2430, 2429, 2428, 2426, 2425, 2423, 2420, 2420, 2416, 2411, 2411, 2407, 2405, 2405, 2403, 2402, 2401, 2401, 2399, 2395, 2395, 2388, 2384, 2382, 2380, 2375, 2372, 2369, 2368, 2357, 2356, 2350, 2342, 2341, 2336, 2336, 2330, 2328, 2318, 2172, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
chess_com_gm_users = ('komodochess', 'vojtechplat', 'magnuscarlsen', 'playmagnus', 'fabianocaruana', 'iuri_shkuro', 'hikaru', 'garrykasparov', 'anand', 'levonaronian', 'firouzja2003', 'gmnakamuracyborg', 'sergeykarjakin', 'lyonbeast', 'grischuk', 'anishgiri', 'lachesisq', 'mamedyarov_twitch', 'gareev-sea', 'gmharikrishna', 'tradjabov', 'anishonyoutube', 'vladimirkramnik', 'sebastian', 'alexonischuk', 'alexander_zubov', 'radzio1987', 'polish_fighter3000', 'yifan0227', 'fairchess_on_youtube', 'hikarunakamura', 'liemle', 'parhamov', 'viditchess', 'exoticprincess', 'shtembuliak', 'cruel_yaro', 'akobian-stl', 'gmvallejo', 'thegadfly1897', 'durarbayli', 'andreikka', 'alexanderl', 'captainjames', 'vladislavkovalev', 'lenderman-ny', 'bigfish1995', 'malev212', 'erenburg-phi', 'zlatan56', 'gmmickeyadams', 'honestgirl', 'kingloek', 'oldbolt', 'dr-bassem', 'bogdandeac', 'chesswy', 'arm-akiba', 'shimanovalex', 'gelashvili-ny', 'kiborg1987', 'kacheishvili-ny', 'zhigalko_sergei', 'pvorontsov', 'illingworth', 'juditpolgar', 'hitaman', 'howitzer14', 'muisback26', 'chepaschess', 'gmcheparinov', 'quesada-mia', 'hedinn', 'mishanick', 'djakovenko', 'psvidler', 'joppie2', 'eltajsafarli', 'yuryshulman', 'nyzhnyk_illia', 'viviania', 'lmcshane', 'verdenotte', 'formerprodigy', 'gm_d85', 'sanan_sjugirov', 'amintabatabaei', 'nigelshort', 'indianlad', 'shankland-ne', 'parimarjan', 'rakhmanov_aleksandr', 'alexander_donchenko', 'oparingrigoriy', 'gguseinov', 'abasovn', 'vmikhalevski', 'hrant_chessmood', 'fireheart92', 'michelangelo', 'senseijulio', 'tigrvshlyape', 'dalmatinac101', 'kempsy', 'liviudieternisipeanu', 'bartoszsocko', 'tesla37', 'kirillshevchenko', 'anton_demchenko', 'newbornnow', 'georgmeier', 'iturrizaga', 'stripunsky-nj', 'alexeishirov', 'rasmussvane', '64genghis64', 'gmvaleduser', 'inopov', 'aggy67', 'vladimir_2020', 'grafdemontekritz', 'grukjr', 'matsenkosergey', 'aryantari', 'tiviakov', 'gmjoel', 'nothedgehog', 'elshan1985', 'sethu0025', 'g3god', 'olomi007', 'gajuchess', 'kaidanov', 'evgenyromanov', 'a-adly', 'fandorine', 'kuzubovyuriy', '1977ivan', 'vovan1991', 'benjamin-nj', 'gmcorrales', 'holt-dal', 'zuraazmai', 'hess-con', 'micki-taryan', 'gmbenjaminbok', 'sandromareco', 'gmarunchess', 'onischuk_v', 'tormoz', 'gmjuliogranda', 'tenismaster', 'chopper1905', 'abhijeetgupta1016', 'vitaliybernadskiy', 'erwinlami', 'duhless', 'gm_huschenbeth', 'aspired', 'dragansolak', 'vegasicilia', 'bykhovsky-stl', 'izoria-man', 'dlugy', 'aivanov-ne', 'sam_chessmood', 'igor_lysyj', 'hustlestandard', 'k_grigoryan', 'hanzo_hasashi1', 'jospem', 'jansmeets', 'nihalsarin', 'gmvar', 'salgadochess91', 'janistantv', 'secret_level', 'jsprepz', 'astralpulse', 'gmg', 'shankland', 'becerra-mia', 'mikhail_antipov', 'gmneiksans', 'dearmike', 'abhijeetgupta', 'matenkatz', 'gm_aag', 'popov_ivan', 'samolo', 'hansen', 'matibar', 'kpio91', 'raunaksadhwani2005', 'champ2005', 'rpragchess', 'bakki78', 'gm_rashid', 'mironius', 'yqperez', 'dr-cro', 'daniel_fridman', 'oleksandr_bortnyk', 'lpsupi', 'pidzsy', 'zaven_chessmood', 'leon009', 'khachiyan-la', 'shield12', 'fishbein-nj', 'saxar85', 'smirnov_pavel', 'eugeneperelshteyn', 'mgeli', 'a-fier', 'evgeny81', 'diwi89', 'gm_crest', 'etserga', 'contora', 'chirila-dal', 'ccgrandmaster', 'paulphoenix87', 'danielnaroditsky', 'gm_mouse_sleep', 'timurgareyev', 'frederiksvane', 'superblysse', 'arambai', 'promen1999', 'atalik', 'obliviate12', 'harsha_bharathakoti', 'vb84', 'dbojkov', 'mickey632702', 'alexvangog', 'tarlev_konstantin', 'littlepeasant', 'gm_arman_pashikian', 'andreystukopin', 'antipov_mikhail_al', 'sergeiaza', 'dudedoel', 'giorgim-bal', 'gopalgn64', 'blefer66', 'molner-arz', 'filippov_anton', 'gmrafpig', 'gulko-nj', 'chadaevnikolay', 'miguelito', 'jim2600', 'bacallao2019', 'gmmarcindziuba', 'gmjoey1', 'kruhor', 'violord', 'aleksandrovaleksei', 'gmmoranda', 'margvelashvili-dal', 'vladimirpotkin', 'vladimirbelous', 'pouya21', 'sipkeernst', 'newzorro', 'humpy1987', 'akshayraj_kore', 'bentsfriend', 'oldweakgm', 'mihailmarin', 'barcenilla-arz', 'janwerle', '1stsecond', 'goryachkina', 'yannickpelletier', 'drvelja', 'gm_buhmann', 'stuwdrey', 'sergiochess83', 'swisspower96', 'gaguna', 'dulemaster', 'alex_goloshchapov', 'renier07', 'pawelczarnota', 'gurevichmikhail', 'perelshteyn-bos', 'susanpolgar', 'kekelidze-con', 'barzed', 'elhelvetico', 's2pac', 'revisor', 'gena217', 'gmvladimir_petkov', 'chessonthemat007', 'carlosalbornoz', 'gmneiksans-bots', 'kosya4ok', 'sychev_klementy', 'vovkcoach', 'emiliocordova', 'gm_barcenilla', 'gmtalks_youtube', 'naroditsky-sf', 'therobot72', 'jcibarra', 'avetik_chessmood', 'krasimir_rusev', 'imtominho', 'kraai-sf', 'dforcen', 'rumpel-vk', 'savelijtartakover', 'robertfontaine', 'potosino1970', 'gukeshdommaraju', 'neurisdr', 'davorinkuljasevic', 'sgchess01', 'superchess02', 'bfinegold-stl', 'gmdrh', 'zvonokchess1996', 'lunaticx', 'gmmelik', 'gmkrikor', 'gordima', 'tranjminh', 'lyndonlistzmendonca', 'koktebel22', 'nijat_a', 'gmartin_petrov', 'anuar_is', 'katerynalagno', 'jfriedel', 'gmbenjaminfinegold', 'bhat-sf', 'gmjankovic', 'maxvavulin', 'xuyinglun', 'roelandpruijssers', 'wizard456', 'gmsrinath', 'gmvladko', 'charbonneau-ny', 'santoblue', 'topotun', 'blagid', 'harutyuniantigrank', 'tal-baron', 'pablolafuente', 'gilmil', 'jessekraai', 'tittanok', 'jrlok', 'jefferyx', 'evgenypigusov', 'paralinch', 'smurfo', 'allanstig', 'elefante33', 'chessqueen', 'gmpacman', 'marianpetrov', 'mnikolov', 'david_arutinian', 'rozum_ivan', 'sosat98', 'rohde-con', 'debashisdebs', 'vi_pranav', 'krikor-mekhitarian', 'morteza_mahjoob', 'kuli4ik', 'diokletian', 'nitzan_steinberg', 'gmianrogers', 'dedic8', 'imre91', 'jlucasvf', 'lvaik', 'gmnickpert', 'wonderfultime', 'jmb2010', 'hungaski-man', 'ruifeng', 'anaconda1983', 'marctarnold', 'mornar1951', 'chesslover0108', 'dulemudule', 'penguingm1', 'genghis_k', 'gmchessrob', 'rodalquilar', 'gmstefanova', 'khalilmousavi98', 'wraku89', 'thecount', 'dusanacns', 'sankalp_gupta2003', 'brightthunder', 'christopheryoo', 'xamax2000', 'trankuilizer', 'courgete', 'gm_batchuluun', 'pittiplatsh', 'raphaei', 'gmmoskalenko', 'gabix_94', 'vladjianu', 'firegrizzly', 'fablibi', 'mauricioflores', 'krush-ny', 'killer-kostya', 'washuntilholes', 'vanea_03', 'nikame', 'markparagua', 'luckytiger', 'artem_ilyin', 'tomskantans', 'mastroskaki', 'michaelq2d5', 'darcylima', 'chowman64', 'diamant-stl', 'deep306', 'paulfromspb', 'speedskater', 'tigergenov', 'vishnuprasanna', 'medvegy', 'karavanela', 'ghandeevam2003', 'drvitman', 'alexrustemov', 'gmcarlosaobregon', 'tetris4jeff', 'azgm', 'rabachess', 'rogapa', 'wanderingknightly', 'actorxu', 'psakhislev', 'konservator69', 'francyim', 'incognito_knight', 'danyuffa', 'vbhat', 'iljazaragatski', 'nrvisakh', 'contrversia', 'maartensolleveld', 'krzyzan94', 'geller_jakov', 'theredking89', 'caefellion', 'eldur16', 'playingsomechess', 'luckyluka04', 'roman_ovetchkin', 'margiriskaunas', 'vyotov', 'frolyanov_d', 'alexyermo', 'federtiu', 'iljushin_alexey', 'prumich', 'andrejs_sokolov', 'hoacin', 'denlaz', 'sahajgrover', 'khamrakulov_djurabek', 'maciek_92', 'rhungaski', 'twannibal', 'romanenko-man', 'geminiboy', 'storne89', 'gmtazbir', 'lastgladiator1', 'hoffentlichreichts', 'fatupss', 'zkid', 'gmpontuscarlsson', 'zablotsky', 'guachiney', 'sumohawk', 'konavets', 'vugarrasulov', 'grigorgrigorov', 'adiosabu', 'hyxiom', 'danir29', 'gonzalez-mia', 'abhidabhi', 'arjun2002', 'alexander_moskalenko', 'tk161', 'mrmotley', 'gm-andreyorlov', 'nguyenduchoa', 'aryangholami', 'julio_becerra', 'arvika', 'georgescu_tiberiu', 'koksik_13', 'anurag328', 'arencibia', 'mikaelyanarman', 'chessdude', 'naaas1', 'liamvrolijk', 'gm_petrovich', 'gm_shahin', 'novendrapriasmoro', 'bocah_sakit', 'misapap', 'gutovandrey', 'kevin_goh', 'amirreza_p', 'gmoratovsky', 'mrkvoz', 'mauriceashley', 'molton', 'oldfish64', 'barys1974', 'robinswinkels', 'ginger_gm', 'thebigboss04', 'arthurkogan', 'oskariot', 'thegormacle', 'faustregel', 'ladheezwousharounm', 'keitharkell', 'justantan', 'holdenhc', 'flycont', 'elgasanov', 'irochka83', 'bankrotten', 'ratkovic_miloje', 'giantslayer83', 'gmmatamoros', 'yuriy88', 'gasconjr', 'dmitrymaximov', 'emilanka', 'anka-sea', 'gm_igor_smirnov', 'kazan28', 'norival', 'parsifal7', 'nakachuk', 'mikolka1985', 'david_klein', 'bilelou', 'attack2mateu', 'shevson', 'fivetrick', 'zubridis', 'thamizhan', 'gm_gonzalezgarcia', 'i_enchev', 'gundavaa89', 'gmkomarov', 'elegance_riks', 'potap1923', 'rd4ever', 'sadikhov_ulvi', 'drenchev', 'luckismyskill', 'chernobay_artem', 'vserguei', 'algeriano22', 'gaizca', 'tartody', 'johansalomon', 'swayamsm', 'lcfox', 'felixblohberger', 'kastanieda_georgui', 'buffy7', 'sevian-bos', 'oskychess', 'ermito2020', '731291', 'arnaudovp', 'snavr', 'kaydentroffchess', 'gmtodorov', 'gmszabo', 'smtheweirdone', 'e-grivas', 'hasangatin_ramil', 'zabdumalik', 'vasily_papin', 'adham_fawzy', 'ghertneck', 'maksat-94', 'hovik_hayrapetyan', 'ss003', 'iskusnyh', 'axves', 'chessialist', 'draskovicl', 'thalaiva', 'khaz_modan', 'regular_legs', 'mikhail_bryakin', 'miromiljkovic', 'gmashley', 'volodar_murzin', 'vrolijk', 'gabrielian_artur', 'sergey_kasparov', 'zaur-mammadov', 'ziachess74', 'ptrajkovic', 'virgo15', 'nigeld', 'homayoontofighi', 'mikhail_golubev', 'skzt', 'daggigretarsson', 'pavel_skatchkov', 'aryamabreu', 'bacyanide', 'keene', 'bilguunsu', 'tdf98pantani', 'knvb', 'aygehovit1992', 'rybka1989', 'hunga', 'arkonada50', 'cyrilmarcelin', 'gmreniergonzalez', 'hissha', 'andyrodri', 'drycounty', 'vincentmasuka', 'danieldardha2005', 'jjosu', 'chabanon', 'dudaki', 'fcchelsea14', 'dibyendu_barua', 'micheliavelli', 'dyadya81', 'margency', 'borisk62', 'kouzari', 'sundarkidambi', 'gevorg_harutjunyan', 'aradej', 'peretyatko', 'fighterman91', 'strelec64', 'vanina1989', 'gmedmundo', 'citychess_said', 'viestur21', 'harutjunyan-gevorg', 'momchilpetkov05', 'petrvelicka', 'ulvi95', 'monmoransy', 'yamalchess', 'gadimbaylia', 'dj_haubi', 'indian-elephant', 'classyel', 'higescha', 'reutik', 'kosanovicgoran', 'b-unit', 'lkaufman-bal', 'mikhail646464', 'davidmarkosyan', 'evgeniy_88', 'ibrayevnb', 'mikheil_kekelidze', 'wrongking', 'szfinx11', 'ketigrant', 'sharapovevgeny', 'vanych', 'riface11', 'evgenysharapov', 'kintoho', 'profylax', 'lilleper1', 'sevenstar82', 'koningsbaars', 'sergey071164', 'checkbits', 'abbasifarhassan', 'mamey78', 'khatanbaatarbazar', 'joshuachick', 'papullon', 'vitarasik', 'art_vandelay1980', 'chessnerdstv', 'youman2016', 'mossedivertenti', 'bubacik', 'slackadaisacal', 'blindfoldkrikor', 'kingloek12345', 'faqade', 'cody1983', 'draganbarlov', 'maximdlugy', 'bobbyhall222', 'gmlazarobruzon', 'son_of_sultan', 'vladimir_zakhartsov', 'chessonado', 'mumbaimover64', '64aramis64', 'praveenb2002', 'deviatkinandrey', 'megazz', 'atomrod', 'particle-accelerator', 'stanislavnovikov', 'l3on4rdo', 'tamirnabaty', 'magicmaster17', 'terorptica', 'nguyenxi', 'rogelio_barcenilla', 'raceking', 'uzbektiger1963', 'udysseus1', 'rightarmoftheforbiddenone', 'yobdc', 'hotshot008', 'shimastream', 'kaffeehaus1851', 'movebymove', 'desconocido04', 'blindginger', 'sabinobrunello', 'markusragger', 'ym-1', 'milanzajic', 'rheticus', 'brandonjacobson', 'ilia_jude', 'ntdvan12', 'ok97', 'blindtakes', 'realpascalc', 'youngkid', 'alexchess1984', 'colerranmorad', 'nemegejas', 'bosiocic', 'attilaczebe', 'ni-hua', 'godspeed', 'pursuitofhappyness2', 'akshatchandra', 'almo64', 'swapchess90', 'man-chuk', 'messiboca', 'uandersson2000', 'drtancredi', 'bewaterbl', 'yardird', 'coldplace', 'jatakk', '4thd-alpeacefulmoon', 'mayachiburdanidze', 'bancsoo', 'resentiment18', 'cipy77', 'gmzaven', 'transponster', 'rookoco', 'belabadea21', 'gm_dmitrij', 'naarciisoo', 'itachiuchiha999', 'evgenyt', 'heliopsis', 'robert_chessmood', 'allstarserious', 'viziruk', 'arcymisie', 'abykhovsky', 'milos021', 'quemirasboboandaparaalla', 'jayg9', 'oursemare', 'tjychess', 'antonsmirnov', 'mathiaswomacka', 'underdogchss', 'babegroot', 'chipsbeer', 'jt', 'macmolner', 'feuerwehrmannsam', 'antonshomoev', 'devikbronstein', 'ftacnikl', 'duanlian', '64dartagnan64', 'evandro_barbosa', 'mrbean', 'lukas_111111', 'megazz66', 'tokyovice', 'ehsan_ghaemmaghami', 'rikikits', 'slipak', 'jacobrafael', 'tashik', 'sjandy', 'giantthresher', 'medallion2028', 'daro94', 'aleksa74', 'gmmikiki', 'scrapnel', 'spiritofwonder', 'gmalsayed', 'mrtattaglia', '124chess', 'el-marmalade', '64pyrrhus64', 'californiadreamgirl', 'chessking518', 'vsanduleac', 'gmtopmoves', 'crescentmoon2411', 'kuybokarovtemur', 'horizonevent', 'goingkronos', 'gmtamaz', 'pomegranate988', 'gmcristhian', 'sigurvegari', 'arseniy_nesterov', 'maghalashvili', '64porthos64', 'tiredofdancing', 'rayrobson', 'asadlivugar', 'thatsallshewrote', 'andrbaryshpolets', 'luiseqp', 'chessbrah', 'alpacacheck', 'brt6209', 'etoilegeniale', 'zstardust', 'levgutman', 'enajer77', 'gm_levan_pantsulaia', 'beca95', 'thej11', 'vorenus_lucius', 'azikom', 'lastreetcmamaison', 'zhuu96', 'liuyuanzhanfu', 'psycho_cowboy', 'charlatante', 'frankieyale', 'pablozarnicki', 'dhoine', 'gorato', 'igmjeroenpiket', 'nowayjosey', 'off-war', 'chessasuke', 'prettyprincess2002', 'ronwhenley', 'gmalex1971', 'mitrabhaa', 'chessintuit', 'msolleveld', 'karpovs_maid', 'gmakobianstl', 'bmw326', 'pepecuenca', 'mirman90', 'bobanbogosavljevic', 'locotito77', 'teamchesscomes', 'notyetfinish', 'tregubovp', 'starworld123', 'eljanov', 'gavrikov', 'taxidermist', 'marcnd', 'detskisad', 'milosji', 'jumbo', 'vidityt', 'olkaa', 'gmasterkiller', 'viih_sou', 'fedoseevvladimir', 'chessathletic', 'beebucks', 'valger', 'niaz1966', 'thipsay', 'untouchablez', 'varuzhanakobian', 'gmjacobaagaard', 'elliotstabler', 'chesstrainer2042', 'daanbrandenburg', 'boorrj', 'huanghou', 'starmatecc', 'kingofconvenience', 'uurknightmare', 'setandgame', 'gmsakk', 'brainwolf', 'stephendedalus83', 'likemachine', 'davidbaramidze', 'fizmat_64', 'cerdasbrs', 'zubov_on_youtube', 'conquering', 'pavelmaletin', 'djojua', 'rhinobritos', 'tptagain', 'zaczel', 'unclejambit', 'nemofishy', 'dragonflow', 'dinopopuslic', 'gm_darwinlaylo', 'markowskitomasz', 'baku_boulevard', 'matthias_wahls', 'snake71', 'peng_li-min', 'gmaxel23', 'anna_muzychuk', 'juancglez', 'fosti19', 'mihaisuba', 'bocharovd', 'm_godena', 'chesseducation4u', 'argyllprince', 'anthonywirig', 'phantom671', 'samhawkins', 'zhuchenchess', 'bestcoacheverr', 'nezuko432', 'erikblomqvist', 'josmito', '64arthos64', 'taekwondoking', 'lestofante90', 'michaelroiz', 'hoangpeony', 'kampkatten', 'queenjudit', 'chito89', 'gats1', 'mirceaparligras', 'sic_transit_gm', 'rfelgaer', 'hesham-a', 'chesstraining77', 'gusanchutik', 'givesalltheflow', 'relaxx87', 'hansontwitch', 'chessfatbear', 'ralf61', 'unseeinghammer', 'byniolus', 'gmbellicose', 'raspadsistema', 'gigantell0', 'daat', 'fartobus', 'niktlt', 'martinez1985', 'jimlahey', 'platy3', 'cemilcan', '123lt', 'zubov_alexander', 'tigrangharamian', 'armenianlion', 'predatoru', 'uchihovich', 'pkclub', 'gmvolokitin', 'zona1', 'grandmaster1369', 'elefante38', 'tjallkompall', 'nesterovsky', 'gnowreopard', 'mr_dynamic', 'cruise999', 'handszar', 'kuchma777', 'agser', 'goanchesstrio', 'volodja49', 'datfifaplaya', 'gabuzyan_chessmood', 'tetrix83', 'clevercrab', 'alexhuzman', 'farout1364', 'firehand33', 'harry_rakkel', 'rolanddeschain', 'azerichess', 'garcho08', 'dogsofwar', 'bryansmith', 'jantimman', 'muradore', 'airgun1', 'igrok3069', 'stupid11', 'beastslovetheprocess', 'lupul_cenusiu', 'drags95', 'drblinditsky', 'igorkovalenko', 'augustomatraga', 'shiirav', 'hellmasterpiece', 'gm_robertomogranzini', 'shyam', 'sharpchess22', 'bluewizzard', 'afgano29', 'batolovic', 'chesswarrior7197', 'bui2512', 'speedoflight0', 'gmcyborek', 'enki007', 'farrukh_amonatov', 'misi_95', 'theexpertpatzer', 'gmbarbosachess', 'tornikesanikidze', 'vladislavtkachev', 'ctionescu', 'weng_pogi', 'tigra', 'bonca64', 'gmmarcelkanarek', 'sahpufjunior', 'hugooakland', 'kondylev', 'fondip', 'tomaspolak74', 'playpositional', 'lovevae', 'worldwar66', 'buenaventuravillamayor', 'maximnovik', 'baag', 'verymasculineegirl', 'abhijeetonyoutube', 'mpcandelario', 'tarhercule', 'indianelephant5', 'baby_legs', 'zeeduin', 'sedlak83', 'alex_stripunsky', 'maybe00', 'glek_igor', 'avantage_ru', 'gmhikaruontwitch', 'dropstonedp', 'dostis', 'joellautier', 'fyall777', 'fodorpoti', 'nuidisvulko', 'chessdjw', 'devonlaratt', 'raja3401', 'gmwso', 'chessmood', 'mazetovic', 'alexsur1981', 'outofaces', 'xxysoul6', 'avolodin', 'padenie_zvezd', 'justanything', 'thevish', 'titan013', 'gmmikhailkazakov', 'incrediblebubacik', 'priezvisko', 'aleksey_sorokin', 'quintilianor', 'vladdobrov', 'sergoy', 'tapuah', 'olga_girya', 'daika91', 'alexander-evdokimov', 'tirantes', 'jaguar_92', '64atilla64', 'teamglud', 'elprofesorsergio', 'chessmisic', 'klan1', 'ellipaehtz', 'myfitnesspal', 'alexandr_predke', 'dancingheaven16', 'boo786', 'delhigm', 'wyharga', 'jimmysenyoo', 'lapitch', 'chesswolf1210', 'viiiagra', 'guesschess_game', 'keyser_stark', 'ucitelot', 'remontada2017', 'krakozia', 'pgw-in-sf', 'deadeye10', 'luisibarrach', 'vincentkeymer', 'gmgamil', 'kantorgergely', 'assezde36', 'solingen2020', 'coach_davido', 'lukianoo', 'sargsyan_shant', 'hugospangenberg', 'iwanyu', 'panevezys', 'reinaldovera', 'infernal_xam', 'sychev_on_youtube', 'gmazapata', 'sanitationengineer', 'baki83', 'kira170', 'mrsavic77', 'andabata', 'zq1', 'timofeevar', 'k_a_s_t_o_r', 'sibelephant', 'pabloricardi', 'gracefulyear', 'pollo2016', 'jhellsten', 'gergo85', 'noukii', 'chesspanda123', 'bornio', 'gmhess', 'kiril1965', 'nnepenthes', 'rz0', 'dorian2017', 'vovanchik0991', 'amirbagheri78', 'hercules64', 'dynamikus', 'depulsooo', 'bulolo', 'lucerne82', 'xdps', 'lutfizadeh', 'excellent1st', 'gmjlh', 'sillygoosemonster', 'adaro', 'eldax64', 'milanovicdan', 'kamilek84', 'spicycaterpillar', 'berny643', 'spartakus1975', 'levi70', 'veselintopalov359', 'y0ung_capablanca', 'superetual', 'gmpanno', 'wonderfuldino', 'olimpusmaximus', 'chessweebs', 'injazzstyle', 'abhyak', 'masya1984', 'stanyga', 'miron_sher', 'schachschnecke', 'kaphrep', 'andrei_belozerov', 'thesilmarils', 'german-2826', 'jvferreira', 'nouma78', 'moskvaa', 'alrayyan2022', 'eltaj_safarli', 'ggg', 'lonelyzeng', 'gmalmeida', 'qashashvilialexandre', 'danielking', 'rednova1729', 'genna1943', 'blueshark23', 'janosik', 'phoenix', 'aryashitake', 'zanyglobal', 'mardakerxxx', 'idontknowchess75', 'paulvelten', 'bonamente', 'hoffmana26', 'peterleko', 'chessawp', 'checky2015', 'godzilla099', 'spasskypy', 'olympusmons95', 'gmyarick', 'real_oceanstorm', 'yuri202062', 'lordillidan', 'rdeux42', 'nikolakis2014', 'alexchess1062', 'ariancito1', 'guenplen', 'nemequittepas', 'lupulescuc', 'gserper', 'mesgen', 'hristosbanikas', 'chessinfun2030', 'jonathantisdall', 'aman', 'petardrenchev', 'tgrade', 'klippfiskkjerringa', 'mugzyyy', 'nikotheodorou', 'masmos97', 'the_impost0r', 'pompey831', 'secretgm', 'mrvenky', 'maestroza', 'billiekimbah', 'c-25', 'bazarov04', 'peterheinenielsen', 'mamedyarov_play', 'jpsmo', 'machupichu10', 'gmsergiobarrientos', 'checom', 'crazyboy26', 'arjunerigaisi2003', 'gm_donatello', 'ddevreugt', 'davor_rogic', 'gakashchess', 'cubitas', 'davidarenas1224', 'kiborg95', 'quietsunset', 'petrkostenko', 'rabguaya201', 'pedro_gines', 'gudmundur_kjartansson', 'gmbartek', 'gogieff', 'salem-ar', 'galperinplaton', 'tocras', 'gen-gutman', 'masis68', 'mammadyarov', 'maestrog', 'kargan90', 'bilodeaua', 'kaka666', 'donkeykong3003', 'azeryahu', 'crazy', 'knightmare82', 'justatrickstar', 'jon_speelman', 'younguzbek', 'crackcubano', 'viktor_matviishen', 'karl_2000', 'dumbo32', 'su1704', 'caissa12', 'chessfreak35', 'ftorodent', 'msb2', 'barcelona_guy', 'mishikomchedlishvili', 'bonuspater18', 'newtie4', 'josemourinho2020', 'alefedorov', 'groovykettle', 'mhebden', 'wanghao', 'cayse', 'ragnaroek66', 'cristhiancruzsanchez', 'chessmastaflash88', 'fghsmn', 'joshdaruca', 'sandropozo', 'kovkov', 'severomorskij', 'vica56', 'kingpusher', 'besmall', 'erebuni91', 'avitalbor', 'tranminhpj', 'leeroy275', 'moro182', 'quaternitychess', 'jguerramzllo', 'kromax33', 'loverespectchess', 'wasawarrior1', 'checkmate_107', 'sourisblanche', 'alexcolovic', 'annawel', 'drawdenied-twitch', 'june57', 'nidjatmamedov', 'grzechu96', 'blindlast7samurai', '64leonidas64', 'sergey_erenburg', 'yoredea', 'lovac58', 'zugazuando', 'like962', 'gigaquparadze', 'feliztigriz', 'tribaldancer', 'gahryman', 'cassoulet', 'georgegiorgadze', 'vartemiev', 'prikolica74', 'riazantsev', 'gmantic', 'vaathi_coming', 'dretch', 'adotand', 'hamonde', 'taykagm', 'chessskool', 'frostnova', 'dr_mortimer', 'winividivici', 'laskerisalive', 'juarinovich', 'nat74', 'gliglu', 'shant_sargsyan', 'marcan2b', 'immatt64', 'gm_splinter', 'maxwell1986', 'b-man', 'olegromanishin', 'aquarium76', 'absentzest', 'rlh2', 'juwen', 'h4parah5', 'josmito80', 'piacramling', 'angelogm', 'dinozavr69', 'n4700000', 'tuksz', 'murtas73', 'grandelicious', 'mareanguis', 'lazy_river', 'mkanarek', 'njal28', 'tacticusbd', 'suryaganguly', 'angry_twin', 'ceraunophilmera', 'lionking4321', 'hamisandwich', 'last7samurai', 'wudileige', 'alphaodin', 'temiksmozg', 'colchonero64', 'zadarchess', 'denis_makhnyov', 'defenceboy', 'gmyaz', 'koutsalogo', 'batica1984', 'copperbrain', 'bubi6', 'sadorra-dal', 'gmpeter008', 'lonelyqueen0', 'turkchess1905', 'attackgm', 'onemoreuser', 'didici', 'egorgeroev', 'aoitsukibluemoon', 'gmmitkov', 'gm_andrei_murariu', 'bulldog167', 'ernestofdz', 'olksuna', 'averagerecord', 'lili_ani', 'akobian-sea', 'paul66666', 'evgchess5', 'london_chess_club', 'swapchesspathshala', 'chefshouse', 'mariyagm', 'gmizeta', 'emperor87', 'mandoran', 'som2310', 'lestri', 'zirafa1', 'sxsw2', 'azerichessss', 'srinavasan', 'rexterminator', 'jncool68', 'jesspinkman', 'burbur555', 'sumopork', 'amir-zaibi', 'azzaro25', 'foewa98', 'kobalia', 'inspektor_rex')

""" Endpoints
    1. Player Profile: https://api.chess.com/pub/player/{username}
    - Get additional details about a player in a game.
    2. Titled Players: https://api.chess.com/pub/titled/{title-abbrev}
    - List of titled-player usernames.
    3. Multi-Game PGN Download: https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}/pgn
    - Download a player's games in a month in PGN format.
    4. List of Monthly Archives: https://api.chess.com/pub/player/{username}/games/archives
    - List of monthly archives for a player.
"""


endpoints = {
    'player_profile': 'https://api.chess.com/pub/player/',  # https://api.chess.com/pub/player/{username}
    'titled_players': 'https://api.chess.com/pub/titled/',  # https://api.chess.com/pub/titled/{title-abbrev}
    'multi_game_pgn_download': 'https://api.chess.com/pub/player/',  # https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}/pgn
    'list_of_monthly_archives': 'https://api.chess.com/pub/player/'  # https://api.chess.com/pub/player/{username}/games/archives
}




def get_player_archives(username):
    url = f'https://api.chess.com/pub/player/{username}/games/archives'
    response = requests.get(url)
    archive_links = []
    if response.status_code == 200:
        archives = json.loads(response.content)
        archive_links = list(archive for archive in archives['archives'])
    return archive_links

def get_titled_players():
    all_title_abbrvs = ['GM', 'WGM', 'IM', 'WIM', 'FM', 'WFM', 'NM', 'WNM', 'CM', 'WCM']
    # GM: 1493 accounts
    # IM: 2008 accounts
    # FM: 3307 accounts
    title_abbrvs = ['GM']
    matched_players = []
    for abbrv in title_abbrvs:
        url = endpoints['titled_players'] + abbrv
        response = requests.get(url)
        if response.status_code == 200:
            titled_players = json.loads(response.content)
            matched_players += titled_players['players']
            matched_players.sort()
            print(matched_players)
    print(matched_players)
    return matched_players

def sort_players_based_on_elo(titled_players):
    # 1. Get the elo of each player
    # 2. Sort the players based on elo

    num_workers = 12
    player_elo_pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_player_elo, player) for player in titled_players]
        for future in concurrent.futures.as_completed(futures):
            player_elo_pairs.append(future.result())

    # Sort pairs based on second entry
    player_elo_pairs.sort(key=lambda x: x[1], reverse=True)
    titled_players, titled_players_elo = zip(*player_elo_pairs)
    return titled_players_elo, titled_players

def try_request(url):
    response = requests.get(url)
    trys = 0
    while response.status_code != 200:
        print('Retrying request:', url)
        time.sleep(2)
        response = requests.get(url)
        trys += 1
        if trys > 1000:
            # throw error
            print(f"Error executing chess com request: {url}")
            return None
    return response

def get_player_elo(username):
    username = str(username)
    url = f'https://api.chess.com/pub/player/{username}/stats'
    response = try_request(url)
    if response:
        try:
            data = response.json()
            fide = int(data['fide'])
            return (username, fide)
        except:
            print(f"Error fetching player profile for {username}: {response.status_code}")
            return (username, 0)
    else:
        return (username, 0)








def download_player_games(username):
    username = str(username)
    combined_pgn = b""
    archive_links = get_player_archives(username)
    for idx, archive_link in enumerate(archive_links):
        archive_pgn_link = archive_link + '/pgn'
        pgn_data = download_player_archive(archive_pgn_link)
        if pgn_data:
            combined_pgn += pgn_data
    save_file = os.path.join(config.chess_com_games_dir, username + '.pgn')
    with open(save_file, "wb") as file:
        file.write(combined_pgn)
    file.close()
    return save_file

def download_player_archive(url):
    response = try_request(url)
    print('Downloading:', url)
    if response.status_code == 200:
        return response.content
    else:
        print("Error downloading archive:", url)
        return None

def get_downloaded_users():
    downloaded_users = []
    for file in os.listdir(config.chess_com_games_dir):
        if file.endswith('.pgn'):
            downloaded_users.append(file.split('.')[0])
    return downloaded_users

def download_chesscom_data():

    # 0. Get downloaded users
    downloaded_users = get_downloaded_users()

    # 1. Get titled players
    titled_players = chess_com_gm_users
    # titled_players = titled_players[:10]
    print('Number of titled players:', len(titled_players))

    # 2. Remove already downloaded players
    titled_players = list(set(titled_players) - set(downloaded_users))
    print('Number of titled players to download:', len(titled_players))

    # 3. Sort players based on elo
    titled_players_elo_sorted, titled_players_sorted = sort_players_based_on_elo(titled_players)
    print(titled_players_elo_sorted)
    print(titled_players_sorted)

    # 2. Download their games each in a separate process
    num_workers = 48
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_player_games, title_player) for title_player in titled_players_sorted]
        for future in concurrent.futures.as_completed(futures):
            print('Finished process:', future.result())

    return 0







if __name__ == '__main__':
    download_chesscom_data()
