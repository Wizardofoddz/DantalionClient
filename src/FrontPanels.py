import collections
import time
import tkinter as tk

from PIL import ImageGrab

import HardwareAbstractionLayer as HAL
from ProcessAbstractionLayer import IDEFProcess
from ProcessLibrary import GeneticIntrinsicsCalculatorProcess
from TrainingSupport import ImageGene


class EvolutionMonitorPanel:
    RootWindow = None

    def __init__(self):
        ImageGene.initialze()

        self.top = tk.Toplevel()
        self.top.title("Evolution Monitor")
        EvolutionMonitorPanel.RootWindow = self.top
        self.popDisplay = Population(self.top, 100, 24, 3, 1, 15, 15)
        self.popDisplay.pack(side=tk.TOP)

        f = tk.Frame(self.top)
        self.status_line = tk.Label(f)
        self.status_line.pack(side=tk.LEFT)

        tk.Button(f, text="START", command=self.start_genetic_calibration).pack(side=tk.RIGHT)
        f.pack(side=tk.TOP)

        self.status_changed = False
        self.status_message = ""

        self.top.after(500, self.handle_status_update)

    def start_genetic_calibration(self):
        # we want undistorted
        # if 'UNDISTORT' not in self.get_imager().controller.imagers[self.get_imager().imager_address].processes:

        icp = GeneticIntrinsicsCalculatorProcess('GeneticIntrinsicsCalculatorProcess', True,
                                                 HAL.Controller.Controllers[0].imagers[0],
                                                 HAL.Controller.Controllers[0].imagers[1])
        icp.initialize({})
        self.popDisplay.start_time = time.time()
        s = icp.get_stage("GeneticSolutionSearchStage")
        s.on_breeder_added = self.on_breeder_added
        s.on_parents_selected = self.on_parents_selected
        s.on_cycle_completed = self.on_cycle_completed
        s.on_child_tested = self.on_child_tested
        icp.status_message = self.change_status_message
        IDEFProcess.DataReady.set()
        IDEFProcess.ActiveProcesses['GenCalib'] = icp

    def change_status_message(self, text):
        """
        Replace status line text with the provided text
        :param text:
        :type text:
        :return:
        :rtype:
        """
        self.status_message = text
        self.status_changed = True

    def handle_status_update(self):
        if self.status_changed:
            self.status_line.config(text=self.status_message)
            self.status_changed = False
        self.top.after(500, self.handle_status_update)

    def on_breeder_added(self, chromosome, bestfitness, breeding_pool, genecounts):
        self.popDisplay.update_population_matrix(bestfitness, breeding_pool, genecounts)

    def on_parents_selected(self, leftchrom, rightchrom):
        self.popDisplay.update_parent_markers(leftchrom, rightchrom)

    def on_cycle_completed(self, minfit, leftfit, rightfit, success, parentschanged):
        self.popDisplay.update_step_display(minfit, leftfit, rightfit, success, parentschanged)

    def on_child_tested(self, chromosomedata, mutationcount):
        self.popDisplay.update_child_vector(chromosomedata)
        self.popDisplay.mutationcount = mutationcount


class Population(tk.Frame):

    def __init__(self, parent, popsize, chromsize, hgap, vgap, hcell, vcell, *args, **kwargs):
        tk.Frame.__init__(self, parent, padx=0, pady=0, bd=3, relief=tk.RAISED)
        self.start_time = time.time()
        self.population = None
        self.mutationcount = 0
        self.cycles = 0
        self.generations = 0
        self.required_fitness = 0
        self.poplen = popsize
        self.chromlen = chromsize
        self.hgap = hgap
        self.vgap = vgap
        self.hcell = hcell
        self.vcell = vcell

        self.leftparentchromosome = None
        self.rightparentchromosome = None
        self.leftparentchromosomeindex = None
        self.rightparentchromosomeindex = None

        self.image_display = tk.Canvas(self, width=self.image_width(), height=self.image_height())
        self.image_display.pack(side=tk.TOP)
        self.diversity = 0
        self.min_fitness = 0
        self.max_fitness = 2
        self.cycle_samples = []
        self.cellrects = None
        self.parentrects = None
        self.parentmarkers = None
        self.new_parents = None
        self.old_parents = None
        self.leftparentchromrects = None
        self.rightparentchromrects = None
        self.childchromrects = None
        self.labels_invalid = True
        self.compute_parent_rects()
        self.compute_parent_markers()
        self.compute_gene_rects()
        self.compute_gene_label_rects()
        self.compute_chromosome_rects()
        self.draw_requests = collections.deque()
        self.label_requests = collections.deque()
        self.last_draw_cycle = 0
        self.force_clear_queue = True
        self.after(500, self.handle_draw_requests)

    def grab_image(self):
        x = EvolutionMonitorPanel.RootWindow.winfo_rootx() + self.winfo_x()
        y = EvolutionMonitorPanel.RootWindow.winfo_rooty() + self.winfo_y()
        x1 = x + self.winfo_width()
        y1 = y + self.winfo_height()
        im = ImageGrab.grab().crop((x, y, x1, y1))
        rgb_im = im.convert('RGB')
        rgb_im.save(
            "../CalibrationRecords/CalibrationGenetics/Frame_{:06d}.jpg".format(self.cycles))

    def compute_parent_rects(self):
        self.parentrects = []
        for x in range(0, self.poplen + 1):
            keypt = self.gene_origin(x, self.chromlen - 1)
            origin = (keypt[0] - self.hgap, 0)
            extent = (keypt[0], keypt[1] + self.vcell + self.vgap)
            self.parentrects.append(self.image_display.create_rectangle(origin[0], origin[1], extent[0], extent[1],
                                                                        fill="gray",
                                                                        outline="black"))

    def compute_parent_markers(self):
        self.parentmarkers = []
        for x in range(0, self.poplen):
            keypt = self.gene_origin(x, self.chromlen - 1)
            origin = (keypt[0], keypt[1] + self.vgap + self.vcell)
            extent = (origin[0] + self.hcell, origin[1] + self.vcell)
            self.parentmarkers.append(self.image_display.create_rectangle(origin[0], origin[1], extent[0], extent[1],
                                                                          fill="gray",
                                                                          outline="black"))

    def compute_gene_rects(self):
        self.cellrects = [[0 for x in range(self.chromlen)] for y in range(self.poplen)]
        for x in range(0, self.poplen):
            for y in range(0, self.chromlen):
                origin = self.gene_origin(x, y)
                extent = (origin[0] + self.hcell + 2, origin[1] + self.vcell + 2)
                self.cellrects[x][y] = self.image_display.create_rectangle(origin[0], origin[1], extent[0], extent[1],
                                                                           fill="gray",
                                                                           outline="black")

    def compute_chromosome_rects(self):
        self.leftparentchromrects = [None] * self.chromlen
        self.rightparentchromrects = [None] * self.chromlen
        self.childchromrects = [None] * self.chromlen
        voffset = self.population_image_height() + 1
        needed = self.chromlen * 3
        excess = 100 - needed
        gap = int(excess / 2)
        childoffset = self.chromlen + gap
        self.leftparentchromotext = [None] * self.chromlen
        for x in range(0, self.chromlen):
            origin = self.gene_origin(x, 0)
            origin = (origin[0], voffset)
            extent = (origin[0] + self.hcell + 2, origin[1] + self.vcell + 2)
            self.leftparentchromrects[x] = self.image_display.create_rectangle(origin[0], origin[1], extent[0],
                                                                               extent[1],
                                                                               fill="gray",
                                                                               outline="black")
            l1 = self.image_display.create_text(origin[0] + 4, origin[1] + 6, fill="darkblue", font="Fira 8",
                                                text=" ")
            l2 = self.image_display.create_text(origin[0] + self.hcell / 2 + 3, origin[1] + self.hcell / 2 + 2,
                                                fill="darkblue", font="Fira 8", text=" ")
            self.leftparentchromotext[x] = (l1, l2)

        self.rightparentchromotext = [None] * self.chromlen
        for x in range(0, self.chromlen):
            origin = self.gene_origin(99 - x, 0)
            origin = (origin[0], voffset)
            extent = (origin[0] + self.hcell + 2, origin[1] + self.vcell + 2)
            self.rightparentchromrects[x] = self.image_display.create_rectangle(origin[0], origin[1], extent[0],
                                                                                extent[1],
                                                                                fill="gray",
                                                                                outline="black")
            l1 = self.image_display.create_text(origin[0] + 4, origin[1] + 6, fill="darkblue", font="Fira 8",
                                                text=" ")
            l2 = self.image_display.create_text(origin[0] + self.hcell / 2 + 3, origin[1] + self.hcell / 2 + 2,
                                                fill="darkblue", font="Fira 8", text=" ")
            self.rightparentchromotext[x] = (l1, l2)

        self.childchromotext = [None] * self.chromlen
        for x in range(0, self.chromlen):
            origin = self.gene_origin(childoffset + x, 0)
            origin = (origin[0], voffset)
            extent = (origin[0] + self.hcell + 2, origin[1] + self.vcell + 2)
            self.childchromrects[x] = self.image_display.create_rectangle(origin[0], origin[1], extent[0],
                                                                          extent[1],
                                                                          fill="gray",
                                                                          outline="black")

            l1 = self.image_display.create_text(origin[0] + 4, origin[1] + 6, fill="darkblue", font="Fira 8",
                                                text=" ")
            l2 = self.image_display.create_text(origin[0] + self.hcell / 2 + 3, origin[1] + self.hcell / 2 + 2,
                                                fill="darkblue", font="Fira 8", text=" ")
            self.childchromotext[x] = (l1, l2)
        self.mutationdisprect = self.image_display.create_rectangle(450, self.image_height() - 50, 650,
                                                                    self.image_height() - 30,
                                                                    fill="white",
                                                                    outline="white")

        self.diversitydisprect = self.image_display.create_rectangle(1140, self.image_height() - 50, 1300,
                                                                     self.image_height() - 30,
                                                                     fill="white",
                                                                     outline="white")
        self.mutationdisptext = self.image_display.create_text(480, self.image_height() - 40, fill="darkblue",
                                                               font="Fira 12",
                                                               text="Mutate")
        self.diversitydisptext = self.image_display.create_text(1160, self.image_height() - 40, fill="darkblue",
                                                                font="Fira 12",
                                                                text="Divers")

    def compute_gene_label_rects(self):
        self.celltext = [[0 for x in range(self.chromlen)] for y in range(self.poplen)]
        for x in range(0, self.poplen):
            for y in range(0, self.chromlen):
                origin = self.gene_origin(x, y)
                l1 = self.image_display.create_text(origin[0] + 4, origin[1] + 6, fill="darkblue", font="Fira 8",
                                                    text=" ")
                l2 = self.image_display.create_text(origin[0] + self.hcell / 2 + 3, origin[1] + self.hcell / 2 + 2,
                                                    fill="darkblue", font="Fira 8", text=" ")
                self.celltext[x][y] = (l1, l2)

    def plot_fitness_samples(self):
        csl = len(self.cycle_samples)
        if csl == 0:
            return
        xstep = self.image_width() / csl
        xpos = 0
        pts = []
        self.image_display.create_rectangle(0, self.image_height() - 30, self.image_width(), self.image_height(),
                                            fill="white",
                                            outline="black")
        for x in self.cycle_samples:
            miny = self.image_height() - int(30 * x[0])
            lefty = self.image_height() - int(30 * x[1])
            righty = self.image_height() - int(30 * x[2])
            bred = x[3]
            cycle = x[4]

            minpt = (int(xpos), miny, "green" if bred else "red")
            leftpt = (int(xpos), lefty, "orange")
            rightpt = (int(xpos), righty, "yellow")
            pts.append((minpt, leftpt, rightpt, cycle))
            xpos += xstep
        first = [True] * 3
        optz = [None] * 3
        for ptl in pts:
            for i in range(0, 3):
                width = 2 if i == 0 else 1
                pt = ptl[i]
                if first[i]:
                    optz[i] = pt
                    first[i] = False
                else:
                    self.image_display.create_line(optz[i][0], optz[i][1], pt[0], pt[1], fill=pt[2], width=width)
                    optz[i] = pt
            if ptl[3] and len(self.population) >= self.poplen and optz[0][0] > 250:
                self.image_display.create_line(optz[0][0], self.image_height() - 10, optz[0][0], self.image_height(),
                                               fill="red", width=2)
        msg = "Cycles: {:05d} Gen: {:05d} Min: {:.3f}  Time: {}".format(self.cycles, self.generations,
                                                                        min(self.required_fitness[0],
                                                                            self.required_fitness[1]),
                                                                        self.hms_string())
        self.image_display.create_text(170, self.image_height() - 24, fill="darkblue", font="Fira 12",
                                       text=msg)

        msg = "M: {:7.3f}%".format((self.mutationcount / (self.cycles * 24)) * 100)
        self.image_display.itemconfig(self.mutationdisptext, text=msg)
        msg = 'D: {:7.3f}%'.format(self.diversity * 100)
        self.image_display.itemconfig(self.diversitydisptext, text=msg)

    def hms_string(self):
        sec_elapsed = time.time() - self.start_time
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    def chromosome_index(self, chromosome):
        for x, p in enumerate(self.population):
            test = p[0]
            if test == chromosome:
                return x
        return None

    def update_parent_markers(self, leftchrom, rightchrom):
        self.leftparentchromosome = leftchrom
        self.rightparentchromosome = rightchrom
        self.update_leftparent_vector(leftchrom)
        self.update_rightparent_vector(rightchrom)
        self.leftparentchromosomeindex = self.chromosome_index(self.leftparentchromosome)
        self.rightparentchromosomeindex = self.chromosome_index(self.rightparentchromosome)

        if self.leftparentchromosomeindex is not None and self.rightparentchromosomeindex is not None:
            self.new_parents = [self.leftparentchromosomeindex, self.rightparentchromosomeindex]

    def handle_draw_requests(self):
        if self.cycles == self.last_draw_cycle:
            self.after(500, self.handle_draw_requests)
            return
        self.last_draw_cycle = self.cycles
        if self.population is None:
            self.force_clear_queue = True
            self.after(500, self.handle_draw_requests)
            self.draw_requests.clear()
            self.label_requests.clear()
            return
        if self.force_clear_queue or len(self.draw_requests) > 9000 or len(self.label_requests) > 9000:
            self.draw_requests.clear()
            self.label_requests.clear()
            self.force_clear_queue = False
        # print('draw {} blocks'.format(len(self.draw_requests)))
        while True:
            try:
                r, c = self.draw_requests.popleft()
                self.set_cell(r, c)
            except IndexError:
                break

        # print('draw {} labels'.format(len(self.label_requests)))
        while True:
            try:
                r, t = self.label_requests.popleft()
                self.image_display.itemconfig(r, text=t)
            except IndexError:
                break

        # print('draw parent markers')
        if self.new_parents is not None:
            if self.old_parents is not None:
                self.image_display.itemconfig(self.parentrects[self.old_parents[0]], fill="gray")
                self.image_display.itemconfig(self.parentrects[self.old_parents[0] + 1], fill="gray")
                self.image_display.itemconfig(self.parentrects[self.old_parents[1]], fill="gray")
                self.image_display.itemconfig(self.parentrects[self.old_parents[1] + 1], fill="gray")
                self.image_display.itemconfig(self.parentmarkers[self.old_parents[0]], fill="gray")
                self.image_display.itemconfig(self.parentmarkers[self.old_parents[1]], fill="gray")
            self.image_display.itemconfig(self.parentrects[self.new_parents[0]], fill="green")
            self.image_display.itemconfig(self.parentrects[self.new_parents[0] + 1], fill="green")
            self.image_display.itemconfig(self.parentrects[self.new_parents[1]], fill="green")
            self.image_display.itemconfig(self.parentrects[self.new_parents[1] + 1], fill="green")
            self.image_display.itemconfig(self.parentmarkers[self.new_parents[0]], fill="green")
            self.image_display.itemconfig(self.parentmarkers[self.new_parents[1]], fill="green")
            self.old_parents = self.new_parents
            self.new_parents = None

        # print('draw fitness samples')
        self.plot_fitness_samples()
        # print('save image')
        self.grab_image()
        self.after(500, self.handle_draw_requests)

    def image_width(self):
        return (self.hgap * self.poplen) + (self.hcell * self.poplen)

    def population_image_height(self):
        return (self.vgap * self.chromlen) + ((self.vcell + 1) * self.chromlen)

    def image_height(self):
        return self.population_image_height() + 50

    def gene_origin(self, chromindex, geneindex):
        hgaps = (chromindex + 1) * self.hgap
        vgaps = (geneindex + 1) * self.vgap
        cellx = chromindex * self.hcell
        celly = geneindex * self.vcell
        return hgaps + cellx, vgaps + celly

    def set_cell(self, rect, color):
        self.image_display.itemconfig(rect, fill=color)

    def update_step_display(self, minfit, leftfit, rightfit, success, parentchanged):
        self.cycles += 1
        mnf = (max(0.75, minfit) - 0.75) * 4
        lf = (max(0.75, leftfit) - 0.75) * 4
        rf = (max(0.75, rightfit) - 0.75) * 4
        self.cycle_samples.append([mnf, lf, rf, success, parentchanged])

    def update_child_vector(self, childgenes):
        self.update_vector(self.childchromrects, self.childchromotext, childgenes)

    def update_leftparent_vector(self, childgenes):
        self.update_vector(self.leftparentchromrects, self.leftparentchromotext, childgenes)

    def update_rightparent_vector(self, childgenes):
        self.update_vector(self.rightparentchromrects, self.rightparentchromotext, childgenes)

    def update_vector(self, chromrects, chromtext, genes):
        for i, g in enumerate(genes):
            cs = "#%02x%02x%02x" % (ImageGene.GeneColors[g][0] & 255, ImageGene.GeneColors[g][1] & 255,
                                    ImageGene.GeneColors[g][2] & 255)
            self.draw_requests.append((chromrects[i], cs))
            lbl = ImageGene.GeneColors[g][3]
            linfo = chromtext[i]
            self.label_requests.append((linfo[0], lbl[0]))
            self.label_requests.append((linfo[1], lbl[1]))

    def update_population_matrix(self, minfitness, breeding_pool, genecounts):
        self.generations += 1
        self.labels_invalid = True
        self.population = breeding_pool
        self.required_fitness = minfitness
        if self.leftparentchromosome is not None and self.rightparentchromosome is not None:
            self.leftparentchromosomeindex = self.chromosome_index(self.leftparentchromosome)
            self.rightparentchromosomeindex = self.chromosome_index(self.rightparentchromosome)
            if self.leftparentchromosomeindex is not None and self.rightparentchromosomeindex is not None:
                self.new_parents = [self.leftparentchromosomeindex, self.rightparentchromosomeindex]
        maxfit = 0
        minfit = 99999
        genesum = {}
        mingc = min(genecounts.values())
        maxgc = max(genecounts.values())
        for x, p in enumerate(self.population):
            chromosome = p[0]
            fitness = p[1]
            if fitness[0] < minfit:
                minfit = fitness[0]
            elif fitness[1] < minfit:
                minfit = fitness[1]
            if fitness[0] > maxfit:
                maxfit = fitness[0]
            elif fitness[1] > maxfit:
                maxfit = fitness[1]
            for y in range(0, len(chromosome)):
                g = chromosome[y]
                if g in genesum:
                    genesum[g] = genesum[g] + fitness[0] + fitness[1]
                else:
                    genesum[g] = fitness[0] + fitness[1]
        for k, v in genesum.items():
            v = v / (len(self.population) * 2)
            if v > 1:
                v = 1
            ImageGene.GeneColors[k][1] = int(v * 255)
            if mingc != maxgc:
                gc = genecounts[k]
                gc = int(((gc - mingc) / (maxgc - mingc)) * 255)
                ImageGene.GeneColors[k][2] = gc
        activegenes = set()
        for x, p in enumerate(self.population):
            chromosome = p[0]
            for y in range(0, len(chromosome)):
                g = chromosome[y]
                activegenes.add(g)
                cs = "#%02x%02x%02x" % (ImageGene.GeneColors[g][0] & 255, ImageGene.GeneColors[g][1] & 255,
                                        ImageGene.GeneColors[g][2] & 255)
                self.draw_requests.append((self.cellrects[x][y], cs))
                lbl = ImageGene.GeneColors[g][3]
                linfo = self.celltext[x][y]
                self.label_requests.append((linfo[0], lbl[0]))
                self.label_requests.append((linfo[1], lbl[1]))
        self.min_fitness = int(self.image_width() * minfit)
        self.max_fitness = int(self.image_width() * maxfit)
        self.diversity = len(activegenes) / len(ImageGene.GeneColors)
