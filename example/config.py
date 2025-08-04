from collections import defaultdict

constellation = {
    "orbits": [{
        "id": 0,
        "eccentricity": 0.0001686,
        "semi_major_axis": 7571000.0,
        "inclination": 87.9166,
        "right_ascension_of_the_ascending_node": 149.4336,
        "argument_of_perigee": 91.4768
    }, {
        "id": 1,
        "eccentricity": 0.0001996,
        "semi_major_axis": 6957000.0,
        "inclination": 141.7312,
        "right_ascension_of_the_ascending_node": 41.6238,
        "argument_of_perigee": 270.7126
    }, {
        "id": 2,
        "eccentricity": 0.0001069,
        "semi_major_axis": 6921000.0,
        "inclination": 43.0037,
        "right_ascension_of_the_ascending_node": 347.161,
        "argument_of_perigee": 271.9912
    }, {
        "id": 3,
        "eccentricity": 0.0001441,
        "semi_major_axis": 6921000.0,
        "inclination": 43.0041,
        "right_ascension_of_the_ascending_node": 104.1028,
        "argument_of_perigee": 262.933
    }, {
        "id": 4,
        "eccentricity": 0.0002935,
        "semi_major_axis": 6921000.0,
        "inclination": 70.0003,
        "right_ascension_of_the_ascending_node": 22.0033,
        "argument_of_perigee": 264.44
    }, {
        "id": 5,
        "eccentricity": 0.0001275,
        "semi_major_axis": 6921000.0,
        "inclination": 53.2154,
        "right_ascension_of_the_ascending_node": 188.4562,
        "argument_of_perigee": 93.2803
    }, {
        "id": 6,
        "eccentricity": 0.0003419,
        "semi_major_axis": 6879000.0,
        "inclination": 97.4975,
        "right_ascension_of_the_ascending_node": 117.0006,
        "argument_of_perigee": 75.9277
    }, {
        "id": 7,
        "eccentricity": 0.0003428,
        "semi_major_axis": 6921000.0,
        "inclination": 69.9988,
        "right_ascension_of_the_ascending_node": 228.9111,
        "argument_of_perigee": 265.4311
    }, {
        "id": 8,
        "eccentricity": 0.0008533,
        "semi_major_axis": 6921000.0,
        "inclination": 52.9808,
        "right_ascension_of_the_ascending_node": 48.9097,
        "argument_of_perigee": 74.735
    }, {
        "id": 9,
        "eccentricity": 0.0001411,
        "semi_major_axis": 6921000.0,
        "inclination": 53.2184,
        "right_ascension_of_the_ascending_node": 5.8202,
        "argument_of_perigee": 95.8908
    }, {
        "id": 10,
        "eccentricity": 0.0002618,
        "semi_major_axis": 6921000.0,
        "inclination": 70.001,
        "right_ascension_of_the_ascending_node": 330.8129,
        "argument_of_perigee": 255.9309
    }, {
        "id": 11,
        "eccentricity": 0.0001207,
        "semi_major_axis": 6921000.0,
        "inclination": 53.053,
        "right_ascension_of_the_ascending_node": 260.4064,
        "argument_of_perigee": 64.5365
    }],
    "satellites": [{
        "inertia":
        [94.807603, 0.0, 0.0, 0.0, 151.15228, 0.0, 0.0, 0.0, 146.220035],
        "mass":
        52.887,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        0,
        "solar_panel": {
            "direction":
            [0.858352216785911, 0.48468367989381406, -0.16826527384847786],
            "area":
            0.45931289637529726,
            "efficiency":
            0.4746423519382975
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.24443082744809944,
            "power": 5.526336287276626,
            "type": 1
        },
        "battery": {
            "capacity": 18225.60447163559,
            "percentage": 0.7632115771182393
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 410.815,
            "power": 5.146,
            "efficiency": 0.526
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 661.162,
            "power": 6.196,
            "efficiency": 0.56
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 651.09,
            "power": 6.098,
            "efficiency": 0.584
        }],
        "mrp_control": {
            "k": 9.806955,
            "ki": 0.00099,
            "p": 21.884891,
            "integral_limit": 0.000827
        },
        "true_anomaly":
        102.83056452046597,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        0
    }, {
        "inertia":
        [142.430296, 0.0, 0.0, 0.0, 183.408367, 0.0, 0.0, 0.0, 136.763639],
        "mass":
        136.957,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        1,
        "solar_panel": {
            "direction":
            [0.40720392352615414, 0.6592577920128184, 0.6321108513032262],
            "area":
            0.27838483145549275,
            "efficiency":
            0.3305836893787611
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.33503882621869874,
            "power": 1.124677274916053,
            "type": 1
        },
        "battery": {
            "capacity": 29550.755683845495,
            "percentage": 0.1600501541122122
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 491.07,
            "power": 5.163,
            "efficiency": 0.567
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 636.161,
            "power": 5.989,
            "efficiency": 0.555
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 455.233,
            "power": 6.441,
            "efficiency": 0.554
        }],
        "mrp_control": {
            "k": 7.099726,
            "ki": 0.00019,
            "p": 17.908339,
            "integral_limit": 0.000272
        },
        "true_anomaly":
        127.73560763729465,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        1
    }, {
        "inertia":
        [137.297074, 0.0, 0.0, 0.0, 193.401823, 0.0, 0.0, 0.0, 84.671604],
        "mass":
        188.415,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        2,
        "solar_panel": {
            "direction":
            [0.9024864996892121, 0.39342860360790133, 0.175305595295332],
            "area":
            0.2979625315223178,
            "efficiency":
            0.30604085228413513
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.333864828686094,
            "power": 7.670223561381203,
            "type": 1
        },
        "battery": {
            "capacity": 17120.427501100534,
            "percentage": 0.6303805049618486
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 675.611,
            "power": 6.63,
            "efficiency": 0.53
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 568.214,
            "power": 6.237,
            "efficiency": 0.546
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 522.287,
            "power": 6.95,
            "efficiency": 0.501
        }],
        "mrp_control": {
            "k": 6.512994,
            "ki": 0.000685,
            "p": 28.464087,
            "integral_limit": 0.000376
        },
        "true_anomaly":
        203.30487695078537,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        2
    }, {
        "inertia":
        [144.267609, 0.0, 0.0, 0.0, 171.631354, 0.0, 0.0, 0.0, 180.561108],
        "mass":
        176.727,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        3,
        "solar_panel": {
            "direction":
            [0.09623968420469939, 0.9936250438270565, -0.05871282197152374],
            "area":
            0.18460669750753933,
            "efficiency":
            0.1308637773534872
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.2673234900258168,
            "power": 4.325498770110421,
            "type": 1
        },
        "battery": {
            "capacity": 23761.60914834641,
            "percentage": 0.6149860667882874
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 638.348,
            "power": 6.755,
            "efficiency": 0.565
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 438.277,
            "power": 6.816,
            "efficiency": 0.534
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 628.544,
            "power": 5.07,
            "efficiency": 0.582
        }],
        "mrp_control": {
            "k": 6.514315,
            "ki": 0.000285,
            "p": 18.12868,
            "integral_limit": 0.000474
        },
        "true_anomaly":
        120.82485283351666,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        3
    }, {
        "inertia":
        [63.770513, 0.0, 0.0, 0.0, 121.673907, 0.0, 0.0, 0.0, 165.893786],
        "mass":
        157.628,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        4,
        "solar_panel": {
            "direction":
            [-0.5746485977441063, 0.6426102611015085, 0.506785005142993],
            "area":
            0.4595620792858621,
            "efficiency":
            0.2777208120006588
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.4148831223839181,
            "power": 3.529066904078062,
            "type": 1
        },
        "battery": {
            "capacity": 27916.001106083284,
            "percentage": 0.8242372110932901
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 524.008,
            "power": 6.388,
            "efficiency": 0.507
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 410.628,
            "power": 6.533,
            "efficiency": 0.505
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 703.487,
            "power": 5.703,
            "efficiency": 0.599
        }],
        "mrp_control": {
            "k": 9.419233,
            "ki": 0.000768,
            "p": 24.33659,
            "integral_limit": 0.000985
        },
        "true_anomaly":
        291.03014257910894,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        4
    }, {
        "inertia":
        [63.55387, 0.0, 0.0, 0.0, 130.649372, 0.0, 0.0, 0.0, 137.641227],
        "mass":
        180.087,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        5,
        "solar_panel": {
            "direction":
            [-0.4130039568311288, 0.1213083930487403, 0.9026139847231281],
            "area":
            0.4730046705616605,
            "efficiency":
            0.36556635572397134
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.47522847084075126,
            "power": 7.068289721077039,
            "type": 1
        },
        "battery": {
            "capacity": 23821.252868060903,
            "percentage": 0.9032298463708318
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 611.595,
            "power": 5.409,
            "efficiency": 0.529
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 557.812,
            "power": 5.662,
            "efficiency": 0.563
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 555.828,
            "power": 6.062,
            "efficiency": 0.573
        }],
        "mrp_control": {
            "k": 6.78099,
            "ki": 0.000779,
            "p": 20.116612,
            "integral_limit": 0.000298
        },
        "true_anomaly":
        316.67167494741113,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        5
    }, {
        "inertia":
        [119.907069, 0.0, 0.0, 0.0, 187.304337, 0.0, 0.0, 0.0, 99.745868],
        "mass":
        61.676,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        6,
        "solar_panel": {
            "direction":
            [0.4533498347216909, 0.021787917939517495, 0.8910663353475298],
            "area":
            0.4584482226550911,
            "efficiency":
            0.2838462216729942
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.1672273819723299,
            "power": 6.018248320537287,
            "type": 1
        },
        "battery": {
            "capacity": 14432.042516759673,
            "percentage": 0.6787100417996634
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 609.544,
            "power": 5.897,
            "efficiency": 0.597
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 661.286,
            "power": 6.575,
            "efficiency": 0.553
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 533.452,
            "power": 6.502,
            "efficiency": 0.597
        }],
        "mrp_control": {
            "k": 6.705162,
            "ki": 0.00048,
            "p": 26.934804,
            "integral_limit": 0.00074
        },
        "true_anomaly":
        289.1165731302157,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        6
    }, {
        "inertia":
        [165.111336, 0.0, 0.0, 0.0, 153.489488, 0.0, 0.0, 0.0, 138.827416],
        "mass":
        134.819,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        7,
        "solar_panel": {
            "direction":
            [0.06523226644932649, 0.17375887743492632, -0.9826253629570317],
            "area":
            0.3328697286029908,
            "efficiency":
            0.22659390127580772
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.22405008819825112,
            "power": 2.0996836979386004,
            "type": 1
        },
        "battery": {
            "capacity": 22521.031571316613,
            "percentage": 0.8418450231044464
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 483.223,
            "power": 6.488,
            "efficiency": 0.533
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 427.907,
            "power": 5.887,
            "efficiency": 0.567
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 500.625,
            "power": 5.113,
            "efficiency": 0.505
        }],
        "mrp_control": {
            "k": 6.811184,
            "ki": 0.000111,
            "p": 30.730221,
            "integral_limit": 0.000854
        },
        "true_anomaly":
        237.83250102937595,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        7
    }, {
        "inertia":
        [142.36944, 0.0, 0.0, 0.0, 179.621228, 0.0, 0.0, 0.0, 152.530939],
        "mass":
        94.346,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        8,
        "solar_panel": {
            "direction":
            [-0.4516199112597861, 0.5332677876507341, 0.7153075718932342],
            "area":
            0.15153792565946406,
            "efficiency":
            0.20271687906976657
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.4711828067436902,
            "power": 8.275540485636048,
            "type": 1
        },
        "battery": {
            "capacity": 23634.357532785427,
            "percentage": 0.40663168078188483
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 402.867,
            "power": 5.728,
            "efficiency": 0.519
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 469.873,
            "power": 5.074,
            "efficiency": 0.509
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 555.939,
            "power": 6.001,
            "efficiency": 0.523
        }],
        "mrp_control": {
            "k": 8.948598,
            "ki": 0.000736,
            "p": 44.132984,
            "integral_limit": 0.000831
        },
        "true_anomaly":
        204.78396685338691,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        8
    }, {
        "inertia":
        [192.232123, 0.0, 0.0, 0.0, 71.227584, 0.0, 0.0, 0.0, 174.962481],
        "mass":
        104.923,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        9,
        "solar_panel": {
            "direction":
            [-0.05999000311382005, 0.007108746415953933, 0.998173664875405],
            "area":
            0.2622877255726378,
            "efficiency":
            0.1163812735896963
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.26830888077161497,
            "power": 7.474379009253043,
            "type": 1
        },
        "battery": {
            "capacity": 28564.09540207535,
            "percentage": 0.9877407914903132
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 565.45,
            "power": 6.529,
            "efficiency": 0.515
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 410.317,
            "power": 6.504,
            "efficiency": 0.531
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 481.796,
            "power": 5.873,
            "efficiency": 0.525
        }],
        "mrp_control": {
            "k": 8.053995,
            "ki": 3.4e-05,
            "p": 30.766423,
            "integral_limit": 0.00037
        },
        "true_anomaly":
        121.86732777038665,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        9
    }, {
        "inertia":
        [86.957569, 0.0, 0.0, 0.0, 61.006225, 0.0, 0.0, 0.0, 111.019507],
        "mass":
        151.172,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        10,
        "solar_panel": {
            "direction":
            [0.010826210358123306, 0.005119963400092689, -0.9999282870006545],
            "area":
            0.34653378340309265,
            "efficiency":
            0.30423566007946357
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.2664864758187201,
            "power": 4.412817555769955,
            "type": 1
        },
        "battery": {
            "capacity": 18411.410850719352,
            "percentage": 0.5914380199531746
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 412.178,
            "power": 6.225,
            "efficiency": 0.524
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 717.797,
            "power": 5.99,
            "efficiency": 0.536
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 441.583,
            "power": 5.052,
            "efficiency": 0.507
        }],
        "mrp_control": {
            "k": 7.472085,
            "ki": 0.000242,
            "p": 35.462534,
            "integral_limit": 0.000379
        },
        "true_anomaly":
        300.352355855839,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        10
    }, {
        "inertia":
        [198.642585, 0.0, 0.0, 0.0, 178.398503, 0.0, 0.0, 0.0, 172.374277],
        "mass":
        152.963,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit":
        11,
        "solar_panel": {
            "direction":
            [-0.8069969929834382, 0.45287315107474735, -0.379027389975075],
            "area":
            0.18565072384588144,
            "efficiency":
            0.42344196909304166
        },
        "sensor": {
            "enabled": False,
            "half_field_of_view": 0.21607942015642526,
            "power": 8.775369190663458,
            "type": 1
        },
        "battery": {
            "capacity": 21162.626168389033,
            "percentage": 0.27795513313766607
        },
        "reaction_wheels": [{
            "rw_type": "Honeywell_HR12",
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 533.546,
            "power": 6.126,
            "efficiency": 0.599
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": 12.0,
            "rw_speed_init": 537.117,
            "power": 6.883,
            "efficiency": 0.566
        }, {
            "rw_type": "Honeywell_HR12",
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": 12.0,
            "rw_speed_init": 433.432,
            "power": 6.089,
            "efficiency": 0.556
        }],
        "mrp_control": {
            "k": 8.755248,
            "ki": 0.000208,
            "p": 34.258791,
            "integral_limit": 0.000192
        },
        "true_anomaly":
        324.1437601892753,
        "mrp_attitude_bn": [0.0, 0.0, 0.0],
        "id":
        11
    }]
}

# "reaction_wheels": [{
#             "rw_type": "Honeywell_HR12",
#             "rw_direction": [1.0, 0.0, 0.0],
#             "max_momentum": 12.0,
#             "rw_speed_init": 533.546,
#             "power": 6.126,
#             "efficiency": 0.599
#         }

reaction_wheels0 = defaultdict(list)
reaction_wheels1 = defaultdict(list)
reaction_wheels2 = defaultdict(list)

satellites = constellation['satellites']

for satellite in satellites:
    reaction_wheels = satellite['reaction_wheels']
    reaction_wheel0 = reaction_wheels[0]
    reaction_wheel1 = reaction_wheels[1]
    reaction_wheel2 = reaction_wheels[2]

    reaction_wheels0['angular_velocity_init'].append(
        reaction_wheel0['rw_speed_init'])
    reaction_wheels1['angular_velocity_init'].append(
        reaction_wheel1['rw_speed_init'])
    reaction_wheels2['angular_velocity_init'].append(
        reaction_wheel2['rw_speed_init'])
