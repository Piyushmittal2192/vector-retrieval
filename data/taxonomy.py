"""Home-improvement taxonomy used by the synthetic catalog and query generators.

Each product type has attributes (sampled per SKU), feature phrases, uses,
and query synonyms. Brands are fictional. The combination of brand x
attributes x features x model number makes every SKU's text distinct, so
nearest-neighbour results, recall and nDCG mean something.
"""

BRANDS = {
    "power tools": ["Voltara", "Ironforge", "Torqline", "Brakmor", "Kestrel Pro", "Ridgeway"],
    "hand tools": ["Ironforge", "Steelhand", "Ridgeway", "Truecut", "Brakmor"],
    "lighting": ["Brightline", "Lumora", "Northbeam", "Halcyon", "Glowpath"],
    "plumbing": ["Aquaflo", "Rivermill", "Cascade & Co", "Puraline", "Delmont"],
    "electrical": ["Voltsafe", "Circuitry", "Northbeam", "Ampere Works"],
    "storage": ["Stackwell", "Garagemax", "Ironforge", "Holdfast"],
    "outdoor": ["Greenreach", "Yardline", "Voltara", "Kestrel Pro", "Sunfield"],
    "paint": ["Colorway", "Truecoat", "Evershade", "Brushwell"],
    "flooring": ["Plankhaus", "Terrastone", "Evershade", "Oakridge"],
}

T = {
    "power tools": {
        "cordless drill": {
            "attrs": {"voltage": ["12V", "18V", "20V", "40V"], "chuck": ["3/8 in.", "1/2 in."], "kit": ["tool only", "1-battery kit", "2-battery kit"]},
            "features": ["brushless motor", "2-speed gearbox", "LED work light", "keyless chuck", "belt clip", "24 clutch settings", "compact head"],
            "uses": ["drilling wood and metal", "driving deck screws", "assembling furniture", "general home repairs"],
            "synonyms": ["drill driver", "battery drill", "power drill"],
        },
        "hammer drill": {
            "attrs": {"voltage": ["18V", "20V", "40V"], "chuck": ["1/2 in."], "kit": ["tool only", "2-battery kit"]},
            "features": ["hammer mode for masonry", "brushless motor", "side handle", "3-mode selector", "metal chuck"],
            "uses": ["drilling into concrete", "anchoring into brick", "masonry work"],
            "synonyms": ["masonry drill", "concrete drill"],
        },
        "impact driver": {
            "attrs": {"voltage": ["12V", "18V", "20V"], "drive": ["1/4 in. hex"], "kit": ["tool only", "1-battery kit", "2-battery kit"]},
            "features": ["3-speed settings", "brushless motor", "1800 in-lbs torque", "LED ring light", "quick-load chuck"],
            "uses": ["driving long lag screws", "deck building", "framing"],
            "synonyms": ["impact gun", "impact screwdriver"],
        },
        "circular saw": {
            "attrs": {"blade": ["6-1/2 in.", "7-1/4 in."], "power": ["cordless 20V", "corded 15 Amp"]},
            "features": ["electric brake", "bevel up to 56 degrees", "dust blower", "laser guide", "magnesium shoe"],
            "uses": ["cutting plywood sheets", "framing lumber", "rip cuts"],
            "synonyms": ["skill saw", "circ saw"],
        },
        "reciprocating saw": {
            "attrs": {"power": ["cordless 20V", "corded 12 Amp"], "stroke": ["1-1/8 in.", "1-1/4 in."]},
            "features": ["tool-free blade change", "orbital action", "variable speed trigger", "anti-vibration handle"],
            "uses": ["demolition", "cutting pipe", "pruning branches"],
            "synonyms": ["sawzall", "recip saw"],
        },
        "angle grinder": {
            "attrs": {"disc": ["4-1/2 in.", "5 in."], "power": ["cordless 20V", "corded 11 Amp"]},
            "features": ["paddle switch", "adjustable guard", "kickback brake", "side handle"],
            "uses": ["cutting rebar", "grinding welds", "removing rust"],
            "synonyms": ["grinder", "cut-off tool"],
        },
        "random orbit sander": {
            "attrs": {"pad": ["5 in.", "6 in."], "power": ["cordless 20V", "corded 3 Amp"]},
            "features": ["dust collection bag", "hook-and-loop pad", "variable speed dial", "low vibration"],
            "uses": ["sanding cabinets", "refinishing furniture", "smoothing drywall patches"],
            "synonyms": ["orbital sander", "palm sander"],
        },
        "jigsaw": {
            "attrs": {"power": ["cordless 20V", "corded 6 Amp"], "orbital": ["4-position orbital"]},
            "features": ["tool-free blade change", "bevel base", "LED light", "dust blower"],
            "uses": ["cutting curves", "cutting countertop openings", "scroll work"],
            "synonyms": ["jig saw", "scroll saw"],
        },
    },
    "hand tools": {
        "claw hammer": {
            "attrs": {"weight": ["16 oz.", "20 oz.", "22 oz."], "handle": ["fiberglass", "steel", "hickory"]},
            "features": ["magnetic nail starter", "anti-vibration grip", "smooth face", "milled face"],
            "uses": ["framing", "driving nails", "light demolition"],
            "synonyms": ["framing hammer", "nail hammer"],
        },
        "tape measure": {
            "attrs": {"length": ["16 ft.", "25 ft.", "35 ft."], "blade": ["1 in. wide", "1-1/4 in. wide"]},
            "features": ["magnetic hook", "13 ft. standout", "nylon coated blade", "belt clip", "fractional markings"],
            "uses": ["measuring rooms", "layout work", "framing"],
            "synonyms": ["measuring tape", "tape rule"],
        },
        "socket set": {
            "attrs": {"pieces": ["40-piece", "100-piece", "200-piece"], "drive": ["1/4 in. and 3/8 in.", "3/8 in. and 1/2 in."]},
            "features": ["SAE and metric", "72-tooth ratchet", "blow-molded case", "chrome vanadium steel"],
            "uses": ["automotive repair", "machinery maintenance", "furniture assembly"],
            "synonyms": ["mechanics tool set", "ratchet set"],
        },
        "screwdriver set": {
            "attrs": {"pieces": ["6-piece", "10-piece", "32-piece"], "tip": ["magnetic tips", "insulated 1000V"]},
            "features": ["cushion grip handles", "Phillips and slotted", "hex bolster", "storage rack"],
            "uses": ["electrical work", "cabinet hardware", "small repairs"],
            "synonyms": ["screwdrivers", "driver set"],
        },
        "level": {
            "attrs": {"length": ["24 in.", "48 in.", "72 in."], "type": ["box beam", "torpedo", "I-beam"]},
            "features": ["magnetic edge", "shock-absorbing end caps", "plumb vial", "milled edge"],
            "uses": ["hanging shelves", "setting posts", "installing cabinets"],
            "synonyms": ["spirit level", "bubble level"],
        },
    },
    "lighting": {
        "LED shop light": {
            "attrs": {"length": ["2 ft.", "4 ft."], "brightness": ["4000 lumens", "5000 lumens", "8000 lumens"], "color temp": ["4000K", "5000K"]},
            "features": ["linkable up to 8 fixtures", "plug-in or hardwire", "pull chain", "frosted lens"],
            "uses": ["garage lighting", "workshop lighting", "basement lighting"],
            "synonyms": ["garage light", "workshop fixture"],
        },
        "LED flood light": {
            "attrs": {"brightness": ["2000 lumens", "3500 lumens", "5000 lumens"], "sensor": ["motion sensor", "dusk to dawn", "no sensor"], "finish": ["black", "bronze", "white"]},
            "features": ["weather resistant", "adjustable heads", "180 degree detection", "wall or eave mount"],
            "uses": ["driveway security lighting", "backyard lighting", "barn lighting"],
            "synonyms": ["security light", "outdoor flood lamp"],
        },
        "recessed light": {
            "attrs": {"size": ["4 in.", "6 in."], "color temp": ["selectable 2700K-5000K", "3000K"], "pack": ["1-pack", "6-pack", "12-pack"]},
            "features": ["canless slim design", "dimmable", "IC rated", "wet location rated"],
            "uses": ["kitchen ceilings", "living room lighting", "bathroom showers"],
            "synonyms": ["can light", "pot light", "downlight"],
        },
        "ceiling fan": {
            "attrs": {"size": ["44 in.", "52 in.", "60 in."], "finish": ["matte black", "brushed nickel", "white"], "control": ["remote control", "pull chain", "smart wifi"]},
            "features": ["integrated LED light", "reversible motor", "DC motor", "outdoor rated"],
            "uses": ["bedroom cooling", "covered patio", "living room airflow"],
            "synonyms": ["fan with light", "ceiling fan light"],
        },
        "string lights": {
            "attrs": {"length": ["24 ft.", "48 ft.", "96 ft."], "bulb": ["Edison S14", "globe G40"], "power": ["plug-in", "solar"]},
            "features": ["shatterproof bulbs", "weatherproof sockets", "connectable strands", "warm white glow"],
            "uses": ["patio decoration", "backyard parties", "pergola lighting"],
            "synonyms": ["patio lights", "cafe lights", "bistro lights"],
        },
    },
    "plumbing": {
        "kitchen faucet": {
            "attrs": {"finish": ["matte black", "stainless steel", "chrome", "brushed gold"], "style": ["pull-down", "single handle", "touchless"]},
            "features": ["magnetic docking sprayer", "spot resistant finish", "ceramic disc valve", "3 spray modes"],
            "uses": ["kitchen sink upgrade", "prep sinks", "farmhouse sinks"],
            "synonyms": ["sink faucet", "kitchen tap"],
        },
        "bathroom faucet": {
            "attrs": {"finish": ["chrome", "matte black", "brushed nickel"], "holes": ["single hole", "4 in. centerset", "8 in. widespread"]},
            "features": ["pop-up drain included", "water saving aerator", "drip-free cartridge"],
            "uses": ["vanity remodel", "powder room", "bathroom sink replacement"],
            "synonyms": ["vanity faucet", "lavatory faucet"],
        },
        "shower head": {
            "attrs": {"type": ["rain", "handheld", "dual"], "flow": ["1.8 GPM", "2.5 GPM"], "finish": ["chrome", "matte black", "brushed nickel"]},
            "features": ["anti-clog nozzles", "5 spray settings", "tool-free install", "adjustable arm"],
            "uses": ["bathroom upgrade", "low pressure showers", "rinsing pets"],
            "synonyms": ["rain shower", "showerhead"],
        },
        "toilet": {
            "attrs": {"shape": ["elongated", "round"], "height": ["comfort height", "standard height"], "flush": ["1.28 GPF", "dual flush"]},
            "features": ["soft close seat", "powerful flush", "easy clean glaze", "one-piece design"],
            "uses": ["bathroom remodel", "replacing a running toilet", "accessible bathrooms"],
            "synonyms": ["water closet", "commode"],
        },
        "tankless water heater": {
            "attrs": {"fuel": ["electric", "natural gas", "propane"], "flow": ["5.5 GPM", "8 GPM", "11 GPM"]},
            "features": ["endless hot water", "digital temperature display", "wifi monitoring", "condensing design"],
            "uses": ["whole house hot water", "replacing a tank heater", "point of use"],
            "synonyms": ["on demand water heater", "instant water heater"],
        },
    },
    "electrical": {
        "smart plug": {
            "attrs": {"pack": ["1-pack", "2-pack", "4-pack"], "protocol": ["wifi", "matter", "zigbee"]},
            "features": ["energy monitoring", "voice assistant compatible", "schedules and timers", "compact design"],
            "uses": ["automating lamps", "holiday lights", "small appliances"],
            "synonyms": ["wifi outlet", "smart outlet"],
        },
        "extension cord": {
            "attrs": {"length": ["25 ft.", "50 ft.", "100 ft."], "gauge": ["16/3", "14/3", "12/3"], "rating": ["indoor", "outdoor"]},
            "features": ["lighted end", "cold weather flexible", "grounded plug", "abrasion resistant jacket"],
            "uses": ["power tools", "outdoor lighting", "job sites"],
            "synonyms": ["power cord", "outdoor cord"],
        },
        "GFCI outlet": {
            "attrs": {"amperage": ["15 Amp", "20 Amp"], "color": ["white", "light almond", "black"], "pack": ["1-pack", "4-pack"]},
            "features": ["self-test", "tamper resistant", "weather resistant", "LED status indicator"],
            "uses": ["kitchen countertops", "bathrooms", "outdoor receptacles"],
            "synonyms": ["ground fault outlet", "GFI receptacle"],
        },
        "smart light switch": {
            "attrs": {"type": ["dimmer", "on/off", "3-way"], "protocol": ["wifi", "matter", "zigbee"], "color": ["white", "black"]},
            "features": ["no neutral required", "app scheduling", "voice control", "paddle design"],
            "uses": ["smart home upgrade", "hallway lighting", "away mode"],
            "synonyms": ["wifi switch", "smart dimmer"],
        },
    },
    "storage": {
        "garage shelving": {
            "attrs": {"tiers": ["4-tier", "5-tier"], "width": ["36 in.", "48 in.", "72 in."], "material": ["steel", "resin"]},
            "features": ["800 lb. capacity per shelf", "boltless assembly", "adjustable shelves", "rust resistant powder coat"],
            "uses": ["garage organization", "basement storage", "pantry storage"],
            "synonyms": ["storage rack", "shelving unit", "metal shelves"],
        },
        "tool chest": {
            "attrs": {"drawers": ["5-drawer", "9-drawer", "12-drawer"], "width": ["26 in.", "41 in.", "52 in."], "finish": ["red", "black", "stainless"]},
            "features": ["ball bearing slides", "keyed lock", "casters", "liner included", "power strip"],
            "uses": ["mechanic workshop", "garage tool storage", "auto shop"],
            "synonyms": ["tool cabinet", "rolling tool box"],
        },
        "storage tote": {
            "attrs": {"size": ["18 gal.", "27 gal.", "40 gal."], "color": ["black and yellow", "clear", "gray"], "pack": ["1-pack", "4-pack"]},
            "features": ["snap-tight lid", "stackable", "heavy duty handles", "weather resistant"],
            "uses": ["seasonal decorations", "moving", "camping gear"],
            "synonyms": ["storage bin", "plastic tote"],
        },
        "workbench": {
            "attrs": {"length": ["48 in.", "60 in.", "72 in."], "top": ["hardwood", "steel", "butcher block"]},
            "features": ["pegboard back", "built-in power strip", "adjustable height", "drawer"],
            "uses": ["woodworking", "garage projects", "repairs"],
            "synonyms": ["work table", "shop bench"],
        },
    },
    "outdoor": {
        "lawn mower": {
            "attrs": {"power": ["cordless 40V", "cordless 80V", "gas 170cc"], "deck": ["20 in.", "21 in.", "30 in."], "drive": ["push", "self-propelled"]},
            "features": ["mulch bag and side discharge", "foldable handle", "height adjustment", "brushless motor"],
            "uses": ["small yards", "half-acre lawns", "hilly terrain"],
            "synonyms": ["mower", "grass cutter"],
        },
        "leaf blower": {
            "attrs": {"power": ["cordless 40V", "cordless 20V", "gas 25cc"], "airflow": ["450 CFM", "600 CFM", "730 CFM"], "type": ["handheld", "backpack"]},
            "features": ["variable speed trigger", "turbo boost", "low noise", "lightweight"],
            "uses": ["fall leaf cleanup", "clearing driveways", "blowing grass clippings"],
            "synonyms": ["blower", "yard blower"],
        },
        "string trimmer": {
            "attrs": {"power": ["cordless 40V", "cordless 20V", "gas 2-cycle"], "cut": ["13 in.", "15 in.", "17 in."]},
            "features": ["bump feed head", "attachment capable", "edging mode", "straight shaft"],
            "uses": ["edging sidewalks", "trimming fence lines", "tall weeds"],
            "synonyms": ["weed eater", "weed wacker", "line trimmer"],
        },
        "garden hose": {
            "attrs": {"length": ["25 ft.", "50 ft.", "100 ft."], "diameter": ["5/8 in.", "3/4 in."], "type": ["rubber", "expandable", "hybrid"]},
            "features": ["kink resistant", "crush proof couplings", "drinking water safe", "burst rated 500 psi"],
            "uses": ["watering gardens", "washing cars", "filling pools"],
            "synonyms": ["water hose", "hose pipe"],
        },
        "gas grill": {
            "attrs": {"burners": ["3-burner", "4-burner", "5-burner"], "fuel": ["propane", "natural gas"], "finish": ["stainless steel", "black"]},
            "features": ["side burner", "cast iron grates", "electronic ignition", "built-in thermometer"],
            "uses": ["backyard cookouts", "searing steaks", "outdoor kitchens"],
            "synonyms": ["bbq grill", "propane grill"],
        },
    },
    "paint": {
        "interior paint": {
            "attrs": {"sheen": ["flat", "eggshell", "satin", "semi-gloss"], "size": ["1 gal.", "5 gal."], "color": ["ultra white", "agreeable gray", "navy", "sage green", "warm beige"]},
            "features": ["paint and primer in one", "low VOC", "scrubbable", "one coat coverage", "stain blocking"],
            "uses": ["living room walls", "bedroom walls", "ceilings"],
            "synonyms": ["wall paint", "latex paint"],
        },
        "exterior paint": {
            "attrs": {"sheen": ["flat", "satin", "semi-gloss"], "size": ["1 gal.", "5 gal."], "color": ["white", "charcoal", "barn red", "slate blue"]},
            "features": ["mildew resistant", "fade resistant", "self-priming", "all weather"],
            "uses": ["siding", "trim and doors", "fences"],
            "synonyms": ["house paint", "outdoor paint"],
        },
        "paint sprayer": {
            "attrs": {"type": ["airless", "HVLP"], "power": ["corded", "cordless 20V"]},
            "features": ["adjustable flow", "multiple tips", "direct from can suction", "easy cleanup"],
            "uses": ["painting fences", "cabinets", "whole house exteriors"],
            "synonyms": ["paint gun", "spray painter"],
        },
        "paint roller kit": {
            "attrs": {"nap": ["3/8 in.", "1/2 in.", "3/4 in."], "pieces": ["5-piece", "9-piece"]},
            "features": ["lint free covers", "tray and liners", "extension pole compatible", "shed resistant"],
            "uses": ["smooth walls", "textured ceilings", "decks"],
            "synonyms": ["roller set", "paint roller"],
        },
    },
    "flooring": {
        "luxury vinyl plank": {
            "attrs": {"thickness": ["6 mil", "12 mil", "20 mil"], "color": ["natural oak", "gray ash", "walnut", "white oak"], "install": ["click lock", "glue down"]},
            "features": ["100% waterproof", "scratch resistant", "attached underlayment", "realistic wood grain"],
            "uses": ["kitchens", "basements", "bathrooms"],
            "synonyms": ["LVP flooring", "vinyl plank floor"],
        },
        "laminate flooring": {
            "attrs": {"thickness": ["8 mm", "10 mm", "12 mm"], "color": ["hickory", "gray oak", "espresso"]},
            "features": ["water resistant", "AC4 wear rating", "beveled edges", "pet friendly"],
            "uses": ["living rooms", "bedrooms", "hallways"],
            "synonyms": ["laminate floor", "wood look laminate"],
        },
        "porcelain tile": {
            "attrs": {"size": ["12 in. x 24 in.", "24 in. x 48 in.", "3 in. x 12 in."], "finish": ["matte", "polished"], "look": ["marble look", "concrete look", "wood look"]},
            "features": ["frost resistant", "rectified edges", "slip resistant", "through-body color"],
            "uses": ["bathroom floors", "shower walls", "patios"],
            "synonyms": ["floor tile", "wall tile"],
        },
        "carpet tile": {
            "attrs": {"size": ["18 in. x 18 in.", "24 in. x 24 in."], "color": ["charcoal", "beige", "blue"], "pack": ["10-pack", "16-pack"]},
            "features": ["peel and stick", "stain resistant", "commercial grade", "cushion backing"],
            "uses": ["basements", "home offices", "playrooms"],
            "synonyms": ["carpet squares", "modular carpet"],
        },
    },
}

ADJECTIVES = ["heavy duty", "compact", "professional", "premium", "lightweight", "contractor grade", "everyday", "high performance"]


def all_types():
    for category, types in T.items():
        for type_name, spec in types.items():
            yield category, type_name, spec
