let CPM = require('/home/guus/Uni/AI_Master/Years/1/sem2/NatCo/aristoo/artistoo/build/artistoo-cjs.js')
var fs = require('fs')

let min_cell_volume = 200
let max_cell_volume = 200
let cell_volume_step = 50

let min_cell_density = 2
let max_cell_density = 3
let cell_density_step = 1

let min_barrier_density = 1
let max_barrier_density = 1
let barrier_density_step = 1

let field_size_x = 210
let field_size_y = 210

let filename = "Assignment_4/js/sim_results/results"+(new Date())+".csv"
fs.appendFile(filename, "cell_volume, cell_density, barrier_density, mean_speed, mean_xdir, mean_ydir \n", function (err) {
    if (err) throw err;
    console.log('Saved!');
})

for (var cell_volume = min_cell_volume; cell_volume <= max_cell_volume; cell_volume += cell_volume_step) {
    for (var cell_density = min_cell_density; cell_density <= max_cell_density; cell_density += cell_density_step) {
        for(var barrier_density = min_barrier_density; barrier_density <= max_barrier_density; barrier_density += barrier_density_step){
            let i = 0

            var posold = {}
            var posnew = {}
            var speeds = []
            var dirXs = []
            var dirYs = []
            
            let config = {
                ndim: 2,
                field_size: [field_size_x, field_size_y],
                conf: {
                T: 20,
                torus: [true, true],
                J: [
                    [NaN, 0, 0],
                    [0, 20, 0],
                    [0, 20, 0],
                ],
                LAMBDA_V: [0, 20, 0], // VolumeConstraint importance per cellkind
                V: [0, cell_volume, 1], // Target volume of each cellkind
                IS_BARRIER: [false, false, true],
                LAMBDA_P: [0, 2, 2], // PerimeterConstraint importance per cellkind
                P : [0,7*Math.sqrt(Math.PI*cell_volume),20],			
                LAMBDA_ACT: [0, 400, 400], // ActivityConstraint importance per cellkind
                MAX_ACT: [0, 20, 0], // Activity memory duration per cellkind
                ACT_MEAN: 'geometric', // Is neighborhood activity computed as a
                },
                simsettings: {
                    logStats:false,
                NRCELLS: [1, 1],
                RUNTIME: 500,
                CELLCOLOR: ['000000', 'FF0000'],
                CANVASCOLOR: 'eaecef',
                ACTCOLOR: [true, false],
                SHOWBORDERS: [true, true],
                zoom: 2,
                },
            }

            let sim = new CPM.Simulation(config,{
                seedCellsInGrid:seedCellsInGrid,
                makeCircularBarriers:makeCircularBarriers,
                initializeGrid:initializeGrid,
                logStats:logStats,
            })

            function seedCellsInGrid() {
                        // Seed the right number of cells for each cellkind
                let step_size_x = Math.floor(this.C.extents[0]/(cell_density+1))
                let step_size_y = Math.floor(this.C.extents[1]/(cell_density+1))
                var i,j
                for(i = 0; i < cell_density+1; i += 1){
                    for(j = 0; j < cell_density+1; j += 1){
                            if(this.C.grid.pixt([i*step_size_x,j*step_size_y]) == 0){
                                this.gm.seedCellAt(1, [i*step_size_x,j*step_size_y])
                            }
                    }
                }
            }


            function logStats(){
                // count the cell IDs currently on the grid:
                let nrcells = 0
                let centroids = this.C.getStat(CPM.CentroidsWithTorusCorrection)
                let totaldist = 0
                let total_xdirection = 0
                let total_ydirection = 0
                for( let i of this.C.cellIDs() ){
                    if (this.C.cellKind(i)==1){
                        if (!(i in posold)){
                            posold[i] = centroids[i]
                            posnew[i] = centroids[i]
                        }else{
                            var difx = posold[i][0]-posnew[i][0]
                            difx += difx<-10?field_size_x:difx>10?-field_size_x:0 // Some evil torus correction
                            var dify = posold[i][1]-posnew[i][1]
                            dify += dify<-10?field_size_y:dify>10?-field_size_y:0 // Please don't look at me
                            let dist = Math.sqrt(Math.pow(difx,2)+Math.pow(dify,2))
                            totaldist+=dist
                            posold[i] = posnew[i]
                            posnew[i] = centroids[i]

                            // here
                            total_xdirection -= difx
                            total_ydirection -= dify
                        }
        
                        nrcells++

                    }
                }
                let meandist = totaldist/nrcells
                let meanxdirection = total_xdirection / nrcells
                let meanydirection = total_ydirection / nrcells
                speeds.push(meandist)
                dirXs.push(meanxdirection)
                dirYs.push(meanydirection)
            }
            

            function makeCircularBarriers() {
                if( !this.helpClasses["gm"] ){ this.addGridManipulator() }
                let step_size_x = Math.floor(this.C.extents[0]/(barrier_density+1))
                let step_size_y = Math.floor(this.C.extents[1]/(barrier_density+1))
                let r = Math.floor(Math.sqrt(cell_volume/(2*Math.PI)))
                let r2 = r*r
                var id_dict = {}

                // Make a loop over a small rectangle	//
                for(let i = -r ; i <= r; i ++){
                    for(let j = -r; j <= r; j ++)
                    {
                        // If there is a point within the radius
                        if (Math.pow(i,2)+Math.pow(j,2) <= r2){
                            // Set every equivalent point in the grid to the correct ID
                            for( let ii = 0; ii<= barrier_density+1; ii ++){
                                for( let jj = 0;jj <= barrier_density+1; jj ++){
                                    if (!([ii,jj] in id_dict)){
                                        id_dict[[ii,jj]] = this.C.makeNewCellID(2)
                                    }
                                    this.C.grid.setpix( [i+ii*step_size_x,j+jj*step_size_y],id_dict[[ii,jj]])
                                }
                            }
                        }
                    }	
                }
            }

            function initializeGrid() {
                if (!this.helpClasses['gm']) {
                    this.addGridManipulator()
                }
                this.makeCircularBarriers()
                this.seedCellsInGrid()
            }

            sim.run()
            let avg_speed = speeds.reduce((a,b) => a+b,0) / (speeds.length)
            let avg_direction_x = dirXs.reduce((a,b) => a+b,0) / (dirXs.length)
            let avg_direction_y = dirYs.reduce((a,b) => a+b,0) / (dirYs.length)
            let newline = cell_volume+","+cell_density+","+barrier_density+","+avg_speed+","+avg_direction_x+","+avg_direction_y+"\n"

            console.log(newline)
            fs.appendFile(filename, newline, function (err) {
                if (err) throw err;
                console.log('Saved!');
            })

        }
    }
}