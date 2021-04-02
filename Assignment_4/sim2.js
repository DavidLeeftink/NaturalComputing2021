let CPM = require('/home/guus/Uni/AI_Master/Years/1/sem2/NatCo/aristoo/artistoo/build/artistoo-cjs.js')
var fs = require('fs')

let min_cell_volume = 100
let max_cell_volume = 100
let cell_volume_step = 50

let min_cell_density = 1
let max_cell_density = 1
let cell_density_step = 1

let min_barrier_density = 2
let max_barrier_density = 2
let barrier_density_step = 1

let avg_speed_dict = {}

for (var cell_volume = min_cell_volume; cell_volume <= max_cell_volume; cell_volume += cell_volume_step) {
    for (var cell_density = min_cell_density; cell_density <= max_cell_density; cell_density += cell_density_step) {
        for(var barrier_density = min_barrier_density; barrier_density <= max_barrier_density; barrier_density += barrier_density_step){

            var posold = {}
            var posnew = {}
            var speeds = []
            let config = {
                ndim: 2,
                field_size: [210, 210],
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
                IS_BARRIER: [false, false, false],
                LAMBDA_P: [0, 2, 2], // PerimeterConstraint importance per cellkind
                P: [0, 200, 20],
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
                logStats:logStats,
                makeCircularBarriers:makeCircularBarriers,
                initializeGrid:initializeGrid,

            })

            function seedCellsInGrid() {
                        // Seed the right number of cells for each cellkind
                let step_size_x = Math.floor(this.C.extents[0]/(cell_density+1))
                let step_size_y = Math.floor(this.C.extents[1]/(cell_density+1))
                var i,j
                for(i = 0; i < cell_density+1; i += 1){
                    for(j = 0; j < cell_density+1; j += 1){
                            if(this.C.grid.pixt([i*step_size_x,j*step_size_y]) == 0){
                                this.gm.seedCellAt(cell_type, [i*step_size_x,j*step_size_y])
                            }
                    }
                }
            }


            function logStats(){
                // count the cell IDs currently on the grid:
                let nrcells = 0
                let centroids = sim.C.getStat(CPM.CentroidsWithTorusCorrection)
                let totaldist = 0
                for( let i of sim.C.cellIDs() ){
            
                    if (sim.C.cellKind(i)==1){
                        if (!(i in posold)){
                            posold[i] = centroids[i]
                            posnew[i] = centroids[i]
                        }else{
                            let difx = posold[i][0]-posnew[i][0]
                            let dify = posold[i][1]-posnew[i][1]
                            let dist = Math.sqrt(Math.pow(difx,2)+Math.pow(dify,2))
                            totaldist+=dist
                            posold[i] = posnew[i]
                            posnew[i] = centroids[i]
                        }
            
                        nrcells++
                    }
            
                }
                let meandist = totaldist/nrcells
                speeds.push(meandist)
                console.log(meandist)
                console.log("\t" + nrcells )
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
                this.seedCellsInGrid()
                this.makeCircularBarriers()
            }

            sim.run()
            avg_speed_dict[[cell_volume,cell_density,barrier_density]] = speeds.reduce((a,b) => a+b,0) / speeds.length
            
        }
    }
}


